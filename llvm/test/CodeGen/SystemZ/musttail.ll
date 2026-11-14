; Test that eligible musttail calls are lowered as sibling calls, just like
; normal tail calls, and that an ineligible tail call that is not marked
; musttail is still lowered as a regular call.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -O0 | FileCheck %s

declare i32 @direct_callee(i32)
declare i32 @five_arg_callee(i32, i32, i32, i32, i32)
declare i32 @vararg_callee(ptr, i32, ...)

define i32 @direct_musttail(i32 %a) {
; CHECK-LABEL: direct_musttail:
; CHECK: jg direct_callee@PLT
  %ret = musttail call i32 @direct_callee(i32 %a)
  ret i32 %ret
}

define i32 @direct_tail(i32 %a) {
; CHECK-LABEL: direct_tail:
; CHECK: jg direct_callee@PLT
  %ret = tail call i32 @direct_callee(i32 %a)
  ret i32 %ret
}

; An indirect musttail call is made through %r1.
define i32 @indirect_musttail(i32 %a, ptr %f) {
; CHECK-LABEL: indirect_musttail:
; CHECK: br %r1
  %ret = musttail call i32 %f(i32 %a, ptr %f)
  ret i32 %ret
}

; The fifth argument is passed in the callee-saved register %r6, so the call
; cannot be lowered as a sibling call. Unlike a musttail call, this is not
; an error; the call is simply lowered as a regular call.
define i32 @ineligible_tail(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
; CHECK-LABEL: ineligible_tail:
; CHECK: brasl %r14, five_arg_callee@PLT
  %ret = tail call i32 @five_arg_callee(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e)
  ret i32 %ret
}

; A varargs function can forward its variable arguments with a musttail call.
; The register save area is spilled and the callee-saved registers restored
; before the jump.
define i32 @vararg_musttail(ptr %p, i32 %x, ...) {
; CHECK-LABEL: vararg_musttail:
; CHECK: stmg %r4, %r15, 32(%r15)
; CHECK: lmg %r6, %r15, 48(%r15)
; CHECK-NEXT: jg vararg_callee@PLT
  %ret = musttail call i32 (ptr, i32, ...) @vararg_callee(ptr %p, i32 %x, ...)
  ret i32 %ret
}
