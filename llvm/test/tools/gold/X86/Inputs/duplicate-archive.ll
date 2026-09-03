target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Weak so that pulling this member from the archive twice (via
; --whole-archive) does not cause a multiple-definition error; the link
; must still succeed, exercising only the duplicate-module code path.
define weak void @foo() {
entry:
  ret void
}
