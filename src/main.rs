//! cce-lock — the cce desktop's session locker.
//!
//! An `ext-session-lock-v1` client: it asks the compositor to lock the
//! session, paints a password prompt on every output, and calls
//! `unlock_and_destroy` only when PAM has accepted the user's credentials.
//!
//! Two properties of the protocol are what make this safe, and both are worth
//! knowing before changing anything here:
//!
//! - **The compositor blanks the session the moment the lock is granted**,
//!   before this process has painted anything. There is no window between
//!   "locked" and "prompt drawn" in which the desktop is visible.
//! - **If this process dies while locked, the session STAYS locked.** cce-fx's
//!   `handle_destroy` (cce-compositor/src/server/lock_manager.rs) deliberately
//!   does not clear the lock state — only the `unlock` request does. So
//!   crashing is a safe failure here, and `kill` is not a bypass. A later
//!   locker can take over an already-locked session; the compositor hands it
//!   `locked` immediately.
//!
//! Which means the dangerous failure is not "it crashed" but "it cannot ever
//! succeed" — a PAM stack that will not start, so no password is ever
//! accepted. [`auth::preflight`] is the guard: the lock is not even requested
//! until PAM has proven it can start.
//!
//! Like cce-cloud, this drives its own event loop and renders through
//! `cce_ui::vk::VkRenderer` rather than implementing cce-ui's `Application`
//! trait — the engine runner creates xdg/layer surfaces, and a lock surface
//! is neither.

mod auth;

use std::collections::HashMap;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_output, delegate_registry, delegate_seat,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    seat::{
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers},
        Capability, SeatHandler, SeatState,
    },
};
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_keyboard, wl_output, wl_seat, wl_surface},
    Connection, Dispatch, Proxy, QueueHandle,
};
use wayland_protocols::ext::session_lock::v1::client::{
    ext_session_lock_manager_v1::ExtSessionLockManagerV1,
    ext_session_lock_surface_v1::{self, ExtSessionLockSurfaceV1},
    ext_session_lock_v1::{self, ExtSessionLockV1},
};

use cce_ui::cosmic_text::{Attrs, Buffer, FontSystem, Metrics, SwashCache};
use cce_ui::scene::layout::Rect;
use cce_ui::vk::{Batch2D, Frame2D, ImageQuad, TextSpan, VkRenderer};
use cce_ui::engine::Vertex;

/// A label queued for the text pass: logical position, size, colour.
struct Label {
    text: String,
    x: f32,
    y: f32,
    size: f32,
    color: [f32; 3],
}

/// One output's lock surface and everything needed to paint it.
struct LockOutput {
    wl_surface: wl_surface::WlSurface,
    lock_surface: ExtSessionLockSurfaceV1,
    renderer: Option<VkRenderer>,
    /// Logical size from the last `configure`; 0 until the first one arrives.
    width: f32,
    height: f32,
    scale: f32,
    /// A buffer may not be attached before the first configure is acked.
    configured: bool,
}

impl Drop for LockOutput {
    fn drop(&mut self) {
        // Swapchain teardown must precede the wl_surface's destruction.
        self.renderer.take();
        self.lock_surface.destroy();
        self.wl_surface.destroy();
    }
}

/// What the UI is doing, which is also what it says on screen.
enum Phase {
    /// Waiting for a password.
    Prompt,
    /// A worker thread is inside PAM. Input is ignored until it answers, so a
    /// held Return cannot queue a hundred attempts against pam_faillock.
    Checking,
    /// PAM accepted; the unlock request has gone out and we are leaving.
    Unlocking,
}

struct AppState {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    compositor_state: CompositorState,

    lock: Option<ExtSessionLockV1>,
    /// Keyed by the wl_output's id, so a surface can be found from either side.
    outputs: HashMap<u32, LockOutput>,
    keyboard: Option<wl_keyboard::WlKeyboard>,

    username: String,
    password: String,
    phase: Phase,
    status: Option<String>,
    caps_lock: bool,
    /// Set once the compositor confirms the session is locked and the previous
    /// contents are hidden.
    locked: bool,
    /// The compositor ended the lock without us asking (`finished`): we must
    /// exit WITHOUT unlocking.
    finished: bool,
    /// PAM accepted before the `locked` event arrived; unlock as soon as it
    /// does. Set only from [`Self::unlock`], which only `AuthEvent::Success`
    /// reaches.
    unlock_when_locked: bool,
    /// `unlock_and_destroy` has been sent. Main must round-trip on this
    /// before exiting.
    unlocked: bool,
    exit: bool,

    font_system: FontSystem,
    swash_cache: SwashCache,
    auth_tx: calloop::channel::Sender<auth::AuthEvent>,
}

impl AppState {
    /// Hand the typed password to PAM on a worker thread. Blocking here would
    /// freeze the lock screen for the length of a faillock delay.
    fn submit(&mut self, qh: &QueueHandle<Self>) {
        if matches!(self.phase, Phase::Checking | Phase::Unlocking) {
            return;
        }
        if self.password.is_empty() {
            self.status = Some("Enter your password".to_string());
            self.draw_all(qh);
            return;
        }
        self.phase = Phase::Checking;
        self.status = None;

        let username = self.username.clone();
        let password = std::mem::take(&mut self.password);
        let ui = self.auth_tx.clone();
        std::thread::spawn(move || {
            let (info_tx, info_rx) = std::sync::mpsc::channel();
            // Pump PAM's running commentary to the screen as it arrives
            // rather than after: a faillock delay can hold `check` for
            // seconds, and the stack explains itself during that time. The
            // sender lives inside the transaction, so this ends on its own
            // when `check` returns.
            let pump_ui = ui.clone();
            let pump = std::thread::spawn(move || {
                while let Ok(ev) = info_rx.recv() {
                    let _ = pump_ui.send(ev);
                }
            });
            let verdict = auth::check(&username, &password, info_tx);
            let _ = pump.join();
            let _ = ui.send(if verdict.is_success() {
                auth::AuthEvent::Success
            } else {
                auth::AuthEvent::Failure { msg: verdict.message() }
            });
            zero(password);
        });
        self.draw_all(qh);
    }

    /// PAM accepted: release the session and go.
    fn unlock(&mut self) {
        // `unlock_and_destroy` before the `locked` event is a PROTOCOL ERROR,
        // and the compositor kills the client for it — leaving the session
        // locked with the locker gone. PAM can answer before `locked` lands
        // (the compositor is still bringing the lock up while the user types
        // into a surface it already configured), so this is reachable.
        if !self.locked {
            log::warn!("authenticated before the locked event; waiting for it");
            self.phase = Phase::Prompt;
            self.status = Some("Locking, one moment…".to_string());
            self.unlock_when_locked = true;
            return;
        }
        let Some(lock) = self.lock.take() else {
            self.exit = true;
            return;
        };
        self.phase = Phase::Unlocking;
        // The ONLY call in this program that opens the session, reached only
        // from `AuthEvent::Success`, which `auth::Verdict::is_success` is the
        // sole producer of.
        lock.unlock_and_destroy();
        // Only now: the protocol says lock surfaces "should be destroyed by
        // the client" AFTER this request, not before.
        self.outputs.clear();
        self.unlocked = true;
        self.exit = true;
    }

    fn create_lock_surface(&mut self, output: &wl_output::WlOutput, qh: &QueueHandle<Self>) {
        let Some(lock) = self.lock.as_ref() else { return };
        let id = output.id().protocol_id();
        if self.outputs.contains_key(&id) {
            return;
        }
        let wl_surface = self.compositor_state.create_surface(qh);
        let lock_surface = lock.get_lock_surface(&wl_surface, output, qh, id);
        self.outputs.insert(
            id,
            LockOutput {
                wl_surface,
                lock_surface,
                renderer: None,
                width: 0.0,
                height: 0.0,
                scale: 1.0,
                configured: false,
            },
        );
    }

    fn draw_all(&mut self, _qh: &QueueHandle<Self>) {
        let ids: Vec<u32> = self.outputs.keys().copied().collect();
        for id in ids {
            self.draw(id);
        }
    }

    /// Paint one output.
    fn draw(&mut self, id: u32) {
        let Some(out) = self.outputs.get(&id) else { return };
        if !out.configured || out.width <= 0.0 || out.height <= 0.0 {
            return;
        }
        let (w, h, scale) = (out.width, out.height, out.scale);

        let (dl, labels) = self.build_scene(w, h);
        let (verts, batches, images, features) = tessellate(&dl, w, h, scale);

        let spans_src: Vec<(Buffer, &Label)> = labels
            .iter()
            .map(|l| (make_text_buffer(&mut self.font_system, &l.text, l.size), l))
            .collect();
        let spans: Vec<TextSpan> = spans_src
            .iter()
            .map(|(buf, l)| TextSpan {
                buffer: buf,
                left: (l.x * scale).round(),
                top: (l.y * scale).round(),
                scale,
                bounds: None,
                default_color: [l.color[0], l.color[1], l.color[2], 1.0],
                rotation: None,
                clip_circle: [0.0; 3],
                clip_extents: [0.0; 2],
            })
            .collect();

        // Split the borrow: the renderer lives in the map, the font system on
        // self, and prepare_text needs both at once.
        let Self { outputs, font_system, swash_cache, .. } = self;
        let Some(out) = outputs.get_mut(&id) else { return };
        let Some(renderer) = out.renderer.as_mut() else { return };
        renderer.prepare_text(font_system, swash_cache, &spans);
        renderer.draw_frame_2d(Frame2D {
            verts: &verts,
            batches: &batches,
            overlay_verts: &[],
            images: &images,
            plate_features: &features,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        });
    }

    /// The lock screen itself: an opaque ground, a centred card, the password
    /// well and its bullets, and one status line.
    fn build_scene(&self, w: f32, h: f32) -> (cce_ui::scene::paint::DisplayList, Vec<Label>) {
        let mut pc = cce_ui::scene::paint::PaintCtx::new();
        let mut labels = Vec::new();

        // Opaque, always. A translucent lock screen would show the desktop it
        // is hiding — the compositor already disabled the normal scene tree,
        // but painting see-through here would still be wrong the moment
        // anything else is composited under it.
        //
        // These channel values are LINEAR, not sRGB: the swapchain is an sRGB
        // format, so the hardware encodes what the shader writes. 0.05 here
        // is #3F3F4B on screen, not the near-black it reads as — which is how
        // this ground first shipped a flat mid-grey. Divide by roughly ten to
        // get the dark you meant; measure with a screenshot, never by eye
        // over the source.
        pc.quad(Rect { x: 0.0, y: 0.0, width: w, height: h }, [0.004, 0.004, 0.006, 1.0]);

        let card_w = 360.0f32.min(w - 40.0);
        let card_h = 170.0f32;
        let card = Rect {
            x: (w - card_w) / 2.0,
            y: (h - card_h) / 2.0,
            width: card_w,
            height: card_h,
        };
        let depth = cce_ui::color::plate_bevel_width();
        pc.plate_spec(&cce_ui::scene::paint::PlateSpec {
            rect: card,
            color: [0.013, 0.013, 0.017, 1.0],
            blur: false,
            window_corners: (true, true, true, true),
            depth,
        });

        labels.push(Label {
            text: self.username.clone(),
            x: card.x + 24.0,
            y: card.y + 22.0,
            size: 15.0,
            color: [1.0, 1.0, 1.0],
        });

        // The password well, rim lit in the highlight the way a focused well
        // is everywhere else in the DE.
        let well = Rect { x: card.x + 24.0, y: card.y + 58.0, width: card_w - 48.0, height: 38.0 };
        pc.quad(well, [0.005, 0.005, 0.007, 1.0]);
        let well_depth = cce_ui::layout::bevel_width().min(well.height * 0.2);
        let hc = cce_ui::color::highlight_primary_color();
        pc.recess_tinted(well, (0.0, 0.0, 0.0, 0.0), well_depth, [hc[0], hc[1], hc[2]]);

        // One dot per character. Never the characters themselves, and never a
        // count in the status line either — both leak the password's length to
        // anyone watching the screen.
        let dot_r = 3.5;
        let dot_gap = 11.0;
        let dots = self.password.chars().count().min(32);
        for i in 0..dots {
            pc.circle(
                well.x + 14.0 + dot_r + i as f32 * dot_gap,
                well.y + well.height / 2.0,
                dot_r,
                [0.80, 0.80, 0.88, 1.0],
            );
        }

        let (status, color) = match self.phase {
            Phase::Checking => ("Checking…".to_string(), [0.72, 0.72, 0.80]),
            Phase::Unlocking => ("Unlocking…".to_string(), [0.72, 0.85, 0.72]),
            Phase::Prompt => match &self.status {
                Some(msg) => (msg.clone(), [0.95, 0.55, 0.55]),
                None if self.caps_lock => ("Caps Lock is on".to_string(), [0.95, 0.80, 0.50]),
                None => (String::new(), [0.55, 0.55, 0.62]),
            },
        };
        if !status.is_empty() {
            labels.push(Label {
                text: status,
                x: card.x + 24.0,
                y: card.y + 112.0,
                size: 12.0,
                color,
            });
        }

        (pc.finish(), labels)
    }
}

/// Best-effort scrub of a password buffer once it has been used.
///
/// Honest about its limits: PAM copies the string into its own allocations and
/// the conversation hands libc a `strdup` of it, and neither is reachable from
/// here. This only clears the copy this process owns, so the window in which a
/// core dump could contain the password is shorter, not closed.
fn zero(mut s: String) {
    unsafe {
        for b in s.as_bytes_mut() {
            *b = 0;
        }
    }
    drop(s);
}

fn make_text_buffer(font_system: &mut FontSystem, text: &str, size: f32) -> Buffer {
    let metrics = Metrics::new(size, size * 1.4);
    let mut buffer = Buffer::new(font_system, metrics);
    let family = cce_ui::layout::control_label_font_parsed().0;
    let attrs = Attrs::new().family(cce_ui::cosmic_text::Family::Name(&family));
    buffer.set_text(font_system, text, attrs, cce_ui::cosmic_text::Shaping::Advanced);
    buffer.shape_until_scroll(font_system, true);
    buffer
}

/// Display list → vertex buffer + renderer batches, converting the
/// tessellator's logical-px clips to physical. Same shape as cce-cloud's.
fn tessellate(
    dl: &cce_ui::scene::paint::DisplayList,
    sw: f32,
    sh: f32,
    scale: f32,
) -> (Vec<Vertex>, Vec<Batch2D>, Vec<ImageQuad>, Vec<[f32; 12]>) {
    let (verts, dl_batches, _dl_images, features) =
        cce_ui::backend::window_runner::tessellate_display_list(dl, sw, sh, scale);
    let batches = dl_batches
        .iter()
        .map(|b| Batch2D {
            scissor: b.scissor.map(|c| {
                (
                    (c.x * scale).max(0.0) as u32,
                    (c.y * scale).max(0.0) as u32,
                    (c.width * scale) as u32,
                    (c.height * scale) as u32,
                )
            }),
            clip_rrect: b
                .clip_rrect
                .map(|c| [c[0] * scale, c[1] * scale, c[2] * scale, c[3] * scale, c[4] * scale]),
            start: b.start,
            end: b.end,
            plate: b.plate,
            blur_behind: b.blur_behind,
        })
        .collect();
    (verts, batches, Vec::new(), features)
}

// ---------------------------------------------------------------------------
// Protocol plumbing
// ---------------------------------------------------------------------------

impl Dispatch<ExtSessionLockManagerV1, ()> for AppState {
    fn event(
        _state: &mut Self,
        _proxy: &ExtSessionLockManagerV1,
        _event: <ExtSessionLockManagerV1 as Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ExtSessionLockV1, ()> for AppState {
    fn event(
        state: &mut Self,
        _proxy: &ExtSessionLockV1,
        event: <ExtSessionLockV1 as Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        match event {
            ext_session_lock_v1::Event::Locked => {
                log::info!("session locked");
                state.locked = true;
                if state.unlock_when_locked {
                    state.unlock_when_locked = false;
                    state.unlock();
                }
            }
            ext_session_lock_v1::Event::Finished => {
                // The compositor refused the lock or ended it. We must exit
                // WITHOUT calling unlock_and_destroy — that request would be
                // a protocol error, and pretending to unlock a session we
                // never locked is not ours to do.
                log::warn!("lock finished by the compositor; exiting without unlocking");
                state.finished = true;
                state.exit = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<ExtSessionLockSurfaceV1, u32> for AppState {
    fn event(
        state: &mut Self,
        _proxy: &ExtSessionLockSurfaceV1,
        event: <ExtSessionLockSurfaceV1 as Proxy>::Event,
        id: &u32,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let ext_session_lock_surface_v1::Event::Configure { serial, width, height } = event {
            let Some(out) = state.outputs.get_mut(id) else { return };
            out.lock_surface.ack_configure(serial);
            out.width = width as f32;
            out.height = height as f32;
            out.configured = true;

            let pw = (out.width * out.scale) as u32;
            let ph = (out.height * out.scale) as u32;
            match out.renderer.as_mut() {
                Some(r) => r.resize(pw, ph),
                None => {
                    out.wl_surface.set_buffer_scale(out.scale as i32);
                    let conn_ptr = _conn.backend().display_id().as_ptr() as *mut std::ffi::c_void;
                    let surf_ptr = out.wl_surface.id().as_ptr() as *mut std::ffi::c_void;
                    out.renderer =
                        Some(unsafe { VkRenderer::new(conn_ptr, surf_ptr, pw, ph, 0.0) });
                }
            }
            state.draw(*id);
        }
    }
}

impl CompositorHandler for AppState {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        let id = self
            .outputs
            .iter()
            .find(|(_, o)| &o.wl_surface == surface)
            .map(|(id, _)| *id);
        let Some(id) = id else { return };
        if let Some(out) = self.outputs.get_mut(&id) {
            out.scale = new_factor as f32;
            out.wl_surface.set_buffer_scale(new_factor);
            if let Some(r) = out.renderer.as_mut() {
                r.resize((out.width * out.scale) as u32, (out.height * out.scale) as u32);
            }
        }
        self.draw(id);
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {}
    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for AppState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, qh: &QueueHandle<Self>, output: wl_output::WlOutput) {
        // A monitor plugged in while locked still gets a prompt rather than
        // the compositor's bare blank.
        self.create_lock_surface(&output, qh);
    }
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        self.outputs.remove(&output.id().protocol_id());
    }
}

impl SeatHandler for AppState {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }
    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Keyboard && self.keyboard.is_none() {
            self.keyboard = self.seat_state.get_keyboard(qh, &seat, None).ok();
        }
    }
    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Keyboard {
            if let Some(kb) = self.keyboard.take() {
                kb.release();
            }
        }
    }
    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl KeyboardHandler for AppState {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
    }
    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
    }

    fn press_key(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        // Everything is ignored mid-check: a held Return would otherwise
        // queue attempts against pam_faillock and lock the account out.
        if matches!(self.phase, Phase::Checking | Phase::Unlocking) {
            return;
        }
        match event.keysym {
            Keysym::Return | Keysym::KP_Enter => {
                self.submit(qh);
                return;
            }
            Keysym::BackSpace => {
                self.password.pop();
                self.status = None;
            }
            Keysym::Escape => {
                // Clears the field. It does NOT dismiss the lock — there is
                // no key that does.
                self.password.clear();
                self.status = None;
            }
            _ => {
                if let Some(text) = event.utf8.as_ref() {
                    for ch in text.chars().filter(|c| !c.is_control()) {
                        self.password.push(ch);
                    }
                    self.status = None;
                }
            }
        }
        self.draw_all(qh);
    }

    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
    }

    fn update_modifiers(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        modifiers: Modifiers,
        _: u32,
    ) {
        if modifiers.caps_lock != self.caps_lock {
            self.caps_lock = modifiers.caps_lock;
            self.draw_all(qh);
        }
    }
}

impl ProvidesRegistryState for AppState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    smithay_client_toolkit::registry_handlers![OutputState, SeatState];
}

delegate_compositor!(AppState);
delegate_output!(AppState);
delegate_seat!(AppState);
delegate_keyboard!(AppState);
delegate_registry!(AppState);

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let username = users::get_current_username()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if username.is_empty() {
        eprintln!("cce-lock: cannot determine the current user; refusing to lock");
        std::process::exit(1);
    }

    // BEFORE locking anything. A PAM stack that will not start would reject
    // every password with the screen already locked, and the only way out
    // would be a TTY and a kill. Failing here costs the user nothing.
    if let Err(e) = auth::preflight(&username) {
        eprintln!("cce-lock: {}", e);
        std::process::exit(1);
    }

    let conn = match Connection::connect_to_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cce-lock: no Wayland connection: {}", e);
            std::process::exit(1);
        }
    };
    let (globals, event_queue) = match registry_queue_init::<AppState>(&conn) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cce-lock: registry init failed: {}", e);
            std::process::exit(1);
        }
    };
    let qh = event_queue.handle();

    let lock_manager: ExtSessionLockManagerV1 = match globals.bind(&qh, 1..=1, ()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cce-lock: compositor does not offer ext-session-lock-v1: {}", e);
            std::process::exit(1);
        }
    };

    let mut event_loop: calloop::EventLoop<AppState> =
        calloop::EventLoop::try_new().expect("event loop");
    let (auth_tx, auth_rx) = calloop::channel::channel::<auth::AuthEvent>();

    cce_ui::scale::set_app_id("cce-lock".to_string());

    let mut state = AppState {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        compositor_state: CompositorState::bind(&globals, &qh).expect("wl_compositor"),
        lock: None,
        outputs: HashMap::new(),
        keyboard: None,
        username,
        password: String::new(),
        phase: Phase::Prompt,
        status: None,
        caps_lock: false,
        locked: false,
        finished: false,
        unlock_when_locked: false,
        unlocked: false,
        exit: false,
        font_system: cce_ui::create_font_system(),
        swash_cache: SwashCache::new(),
        auth_tx,
    };

    state.lock = Some(lock_manager.lock(&qh, ()));
    // Surfaces for the outputs that already exist; later ones arrive through
    // OutputHandler::new_output.
    let outputs: Vec<wl_output::WlOutput> = state.output_state.outputs().collect();
    for output in &outputs {
        state.create_lock_surface(output, &qh);
    }

    event_loop
        .handle()
        .insert_source(auth_rx, |event, _, state| {
            let calloop::channel::Event::Msg(event) = event else { return };
            match event {
                auth::AuthEvent::Success => state.unlock(),
                auth::AuthEvent::Failure { msg } => {
                    state.phase = Phase::Prompt;
                    state.status = Some(msg);
                }
                auth::AuthEvent::Info { msg } => {
                    state.status = Some(msg);
                }
            }
        })
        .expect("auth channel");

    calloop_wayland_source::WaylandSource::new(conn.clone(), event_queue)
        .insert(event_loop.handle())
        .expect("wayland source");

    while !state.exit {
        if event_loop
            .dispatch(std::time::Duration::from_millis(50), &mut state)
            .is_err()
        {
            break;
        }
        // Redraw outside the event handlers: an auth result arrives on the
        // calloop channel with no qh in scope.
        if !matches!(state.phase, Phase::Unlocking) {
            let ids: Vec<u32> = state.outputs.keys().copied().collect();
            for id in ids {
                state.draw(id);
            }
        }
    }

    // A flush is NOT enough after unlock_and_destroy, and the protocol says
    // so outright: without a sync the server may terminate this client before
    // it processes the request, and the session would stay locked with no
    // locker running. Round-trip, then go.
    if state.unlocked {
        if let Err(e) = conn.roundtrip() {
            log::error!("roundtrip after unlock failed: {}", e);
        }
    }
    let _ = conn.flush();
    if state.finished {
        std::process::exit(1);
    }
}
