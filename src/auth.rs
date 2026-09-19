//! PAM credential checking for the locker.
//!
//! Deliberately the smallest thing that can answer "is this the person who
//! owns this session": `pam_authenticate` followed by `pam_acct_mgmt`, and
//! nothing else. In particular there is **no `pam_open_session`** — the
//! session the locker is guarding already exists, and opening a second one
//! from here would run the session stack's side effects (pam_gnome_keyring's
//! `auto_start` forks out of this multi-threaded Vulkan process) for no
//! reason. The greeter hit exactly that and froze on "Authenticating..." with
//! the password already accepted; here the equivalent freeze would leave the
//! screen locked.
//!
//! The verdict is deliberately a plain `bool` chosen in ONE place
//! ([`Verdict::is_success`]), so there is no path to an unlock that did not
//! come from both PAM calls returning SUCCESS.

use std::ffi::{CStr, CString};

/// PAM service name — `/etc/pam.d/cce-lock`, shipped in this crate's `pam/`
/// dir and installed by `ccebuild install-system`.
///
/// A constant, never an environment variable or argument: a locker whose PAM
/// stack can be chosen by its caller is a locker anyone with a shell can point
/// at a permissive service.
pub const PAM_SERVICE: &str = "cce-lock";

/// What the worker thread reports back to the UI.
pub enum AuthEvent {
    /// PAM accepted the credentials. The ONLY value that may unlock.
    Success,
    /// PAM rejected them, or errored. `msg` is for the user, not a log line.
    Failure { msg: String },
    /// A `TEXT_INFO` / `ERROR_MSG` from the stack while it ran — a faillock
    /// delay notice, an expiry warning.
    Info { msg: String },
}

/// Everything the conversation function is allowed to see.
struct ConvData {
    username: String,
    password: String,
    sender: std::sync::mpsc::Sender<AuthEvent>,
}

/// The PAM conversation: answer the password prompt from the buffer, echo the
/// username back for an echoing prompt, and forward anything informational.
///
/// `extern "C"`, so it must not unwind: every fallible conversion below is
/// total.
extern "C" fn converse(
    num_msg: libc::c_int,
    msg: *mut *mut pam_sys::PamMessage,
    out_resp: *mut *mut pam_sys::PamResponse,
    appdata_ptr: *mut libc::c_void,
) -> libc::c_int {
    if appdata_ptr.is_null() || num_msg <= 0 {
        return pam_sys::PamReturnCode::CONV_ERR as libc::c_int;
    }
    let data = unsafe { &*(appdata_ptr as *const ConvData) };

    let resp = unsafe {
        libc::calloc(num_msg as usize, std::mem::size_of::<pam_sys::PamResponse>())
            as *mut pam_sys::PamResponse
    };
    if resp.is_null() {
        return pam_sys::PamReturnCode::BUF_ERR as libc::c_int;
    }

    for i in 0..num_msg as isize {
        unsafe {
            let m = &**msg.offset(i);
            let r = &mut *resp.offset(i);
            let style = m.msg_style;
            if style == pam_sys::PamMessageStyle::PROMPT_ECHO_OFF as libc::c_int {
                // unwrap_or_default, not unwrap: an interior NUL in the typed
                // password would otherwise panic across this extern "C"
                // boundary, which aborts the process — and aborting the
                // locker leaves the session locked with no way in. An empty
                // response just fails the attempt.
                let pass = CString::new(data.password.clone()).unwrap_or_default();
                r.resp = libc::strdup(pass.as_ptr());
            } else if style == pam_sys::PamMessageStyle::PROMPT_ECHO_ON as libc::c_int {
                let user = CString::new(data.username.clone()).unwrap_or_default();
                r.resp = libc::strdup(user.as_ptr());
            } else if !m.msg.is_null() {
                let text = CStr::from_ptr(m.msg).to_string_lossy().into_owned();
                let _ = data.sender.send(AuthEvent::Info { msg: text });
            }
        }
    }

    unsafe { *out_resp = resp };
    pam_sys::PamReturnCode::SUCCESS as libc::c_int
}

/// A PAM transaction, ended on drop so no handle outlives an attempt.
struct Transaction {
    handle: *mut pam_sys::PamHandle,
    last: pam_sys::PamReturnCode,
    // Held alive for as long as PAM holds the pointer into it.
    _data: Box<ConvData>,
}

impl Transaction {
    fn start(username: &str, password: &str, sender: std::sync::mpsc::Sender<AuthEvent>) -> Result<Self, pam_sys::PamReturnCode> {
        let data = Box::new(ConvData {
            username: username.to_string(),
            password: password.to_string(),
            sender,
        });
        let conv = pam_sys::PamConversation {
            conv: Some(converse),
            data_ptr: &*data as *const ConvData as *mut libc::c_void,
        };
        let mut handle: *mut pam_sys::PamHandle = std::ptr::null_mut();
        let rc = pam_sys::start(PAM_SERVICE, Some(username), &conv, &mut handle);
        if rc != pam_sys::PamReturnCode::SUCCESS {
            return Err(rc);
        }
        Ok(Self { handle, last: pam_sys::PamReturnCode::SUCCESS, _data: data })
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { pam_sys::end(&mut *self.handle, self.last) };
        }
    }
}

/// The single place an unlock can be authorized.
pub struct Verdict {
    authenticate: pam_sys::PamReturnCode,
    acct_mgmt: pam_sys::PamReturnCode,
}

impl Verdict {
    /// True only when BOTH PAM calls returned SUCCESS. Every other outcome —
    /// a rejection, an expired account, an internal PAM error, a stack that
    /// could not be started — is a failure, because a locker that opens on
    /// anything it does not understand is not a lock.
    pub fn is_success(&self) -> bool {
        self.authenticate == pam_sys::PamReturnCode::SUCCESS
            && self.acct_mgmt == pam_sys::PamReturnCode::SUCCESS
    }

    /// What to show the user. PAM's own codes are not phrased for a lock
    /// screen, and echoing them leaks stack detail to whoever is standing
    /// there, so a wrong password gets one plain sentence.
    pub fn message(&self) -> String {
        if self.authenticate == pam_sys::PamReturnCode::AUTH_ERR {
            "Incorrect password".to_string()
        } else if self.authenticate != pam_sys::PamReturnCode::SUCCESS {
            format!("Authentication failed ({:?})", self.authenticate)
        } else {
            format!("Account unavailable ({:?})", self.acct_mgmt)
        }
    }
}

/// Check `password` against `username`'s credentials. Blocks — PAM stacks
/// sleep on failure (pam_faillock) — so callers run this on a worker thread.
pub fn check(username: &str, password: &str, sender: std::sync::mpsc::Sender<AuthEvent>) -> Verdict {
    let mut tx = match Transaction::start(username, password, sender) {
        Ok(tx) => tx,
        Err(rc) => {
            log::error!("pam_start({}) failed: {:?}", PAM_SERVICE, rc);
            // Not an unlock: a stack that will not start cannot vouch for
            // anyone. `preflight` exists so this is caught before locking.
            return Verdict { authenticate: rc, acct_mgmt: rc };
        }
    };

    let authenticate = unsafe { pam_sys::authenticate(&mut *tx.handle, pam_sys::PamFlag::NONE) };
    tx.last = authenticate;
    if authenticate != pam_sys::PamReturnCode::SUCCESS {
        return Verdict { authenticate, acct_mgmt: authenticate };
    }

    let acct_mgmt = unsafe { pam_sys::acct_mgmt(&mut *tx.handle, pam_sys::PamFlag::NONE) };
    tx.last = acct_mgmt;
    Verdict { authenticate, acct_mgmt }
}

/// Prove the PAM stack can be started BEFORE the session is locked.
///
/// This is the difference between "the lock did not engage" and "you cannot
/// get back in". Without `/etc/pam.d/cce-lock` installed, `pam_start` fails
/// and every attempt would be rejected — with the screen already locked and
/// the only way out a TTY and a kill. So the locker refuses to lock at all
/// unless this passes.
pub fn preflight(username: &str) -> Result<(), String> {
    // `pam_start` does NOT fail on a missing service file: libpam falls back
    // to /etc/pam.d/other, which is pam_deny on Arch, so the start succeeds
    // and every password is then rejected. That locked the live session out
    // on 2026-09-19. The file itself has to be looked for.
    let installed = ["/etc/pam.d", "/usr/lib/pam.d"]
        .iter()
        .any(|dir| std::path::Path::new(dir).join(PAM_SERVICE).is_file());
    if !installed {
        return Err(format!(
            "PAM service file /etc/pam.d/{} is not installed — every password \
             would be rejected by the `other` fallback. Run: ccebuild install-system",
            PAM_SERVICE
        ));
    }

    let (tx, _rx) = std::sync::mpsc::channel();
    match Transaction::start(username, "", tx) {
        Ok(_) => Ok(()),
        Err(rc) => Err(format!(
            "PAM service {:?} unavailable ({:?}) — is /etc/pam.d/{} installed? \
             Run: ccebuild install-system",
            PAM_SERVICE, rc, PAM_SERVICE
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one gate. If this ever admits a non-SUCCESS pair, the lock opens
    /// on a failed credential check.
    #[test]
    fn only_success_on_both_calls_unlocks() {
        let ok = pam_sys::PamReturnCode::SUCCESS;
        let bad = pam_sys::PamReturnCode::AUTH_ERR;
        let other = pam_sys::PamReturnCode::ABORT;

        assert!(Verdict { authenticate: ok, acct_mgmt: ok }.is_success());
        assert!(!Verdict { authenticate: bad, acct_mgmt: ok }.is_success());
        assert!(!Verdict { authenticate: ok, acct_mgmt: bad }.is_success());
        assert!(!Verdict { authenticate: other, acct_mgmt: other }.is_success());
    }

    #[test]
    fn a_wrong_password_says_so_without_leaking_the_stack() {
        let v = Verdict {
            authenticate: pam_sys::PamReturnCode::AUTH_ERR,
            acct_mgmt: pam_sys::PamReturnCode::AUTH_ERR,
        };
        assert_eq!(v.message(), "Incorrect password");
    }
}
