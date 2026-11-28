; Test that an error is reported when a musttail call cannot be lowered as
; a tail call.
;
; RUN: rm -rf %t && split-file %s %t
; RUN: not --crash llc -mtriple=s390x-linux-gnu < %t/r6.ll 2>&1 | FileCheck --check-prefix=ELF %s
; RUN: not --crash llc -mtriple=s390x-linux-gnu < %t/stackarg.ll 2>&1 | FileCheck --check-prefix=ELF %s
; RUN: not --crash llc -mtriple=s390x-linux-gnu < %t/swifterror.ll 2>&1 | FileCheck --check-prefix=ELF %s
; RUN: not --crash llc -mtriple=s390x-linux-gnu < %t/swiftself.ll 2>&1 | FileCheck --check-prefix=ELF %s
; RUN: not --crash llc -mtriple=s390x-linux-gnu -O0 -fast-isel < %t/r6.ll 2>&1 | FileCheck --check-prefix=ELF %s
; RUN: not --crash llc -mtriple=s390x-ibm-zos < %t/r6.ll 2>&1 | FileCheck --check-prefix=XPLINK64 %s
; RUN: not --crash llc -mtriple=s390x-ibm-zos < %t/simple.ll 2>&1 | FileCheck --check-prefix=XPLINK64 %s

; On Linux, each of the cases below makes the call ineligible for a sibling
; call, so lowering it must fail rather than silently emit a regular call.
;
; ELF: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail

; On z/OS XPLINK64, tail calls are disabled entirely, so a musttail call
; fails regardless of its arguments.
;
; XPLINK64: LLVM ERROR: musttail call is not supported on XPLINK64

;--- r6.ll
; The fifth integer argument is passed in the callee-saved register %r6.
; FIXME: in this particular case, the value in %r6 ends up being the same for caller
;        and callee, so the tail call would actually be possible. This should be fixed
;        once the backend is smarter about which tail calls it permits.
define i32 @caller(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
  %ret = musttail call i32 @callee(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e)
  ret i32 %ret
}

define i32 @callee(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
  ret i32 %a
}

;--- stackarg.ll
; Only four floating-point arguments are passed in registers, so the fifth
; one is passed on the stack.
define double @caller(double %a, double %b, double %c, double %d, double %e) {
  %ret = musttail call double @callee(double %a, double %b, double %c, double %d, double %e)
  ret double %ret
}

define double @callee(double %a, double %b, double %c, double %d, double %e) {
  ret double %e
}

;--- swifterror.ll
; A function with a swifterror argument cannot make any tail call.
define ptr @caller(ptr swifterror %err) {
  %ret = musttail call ptr @callee(ptr swifterror %err)
  ret ptr %ret
}

declare ptr @callee(ptr swifterror %err)

;--- swiftself.ll
; A swiftself argument is passed in the callee-saved register %r10.
define ptr @caller(ptr swiftself %p) {
  %ret = musttail call ptr @callee(ptr swiftself %p)
  ret ptr %ret
}

define ptr @callee(ptr swiftself %p) {
  ret ptr %p
}

;--- simple.ll
define i32 @caller(i32 %a) {
  %ret = musttail call i32 @callee(i32 %a)
  ret i32 %ret
}

define i32 @callee(i32 %a) {
  ret i32 %a
}
