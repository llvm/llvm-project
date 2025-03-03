; RUN: llc %s -o - | FileCheck %s

; Tail calls which have stack arguments in the same offsets as the caller do not
; need to load and store the arguments from the stack.

target triple = "aarch64"

declare void @func(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i32 %j)

; CHECK-LABEL: wrapper_func:
define void @wrapper_func(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i32 %j) {
  ; CHECK: // %bb.
  ; CHECK-NEXT: b func
  tail call void @func(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i32 %j)
  ret void
}

; CHECK-LABEL: wrapper_func_zero_arg:
define void @wrapper_func_zero_arg(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i32 %j) {
  ; CHECK: // %bb.
  ; CHECK-NEXT: str wzr, [sp, #8]
  ; CHECK-NEXT: b func
  tail call void @func(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i32 0)
  ret void
}

; CHECK-LABEL: wrapper_func_overriden_arg:
define void @wrapper_func_overriden_arg(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i32 %j) {
  ; CHECK: // %bb.
  ; CHECK-NEXT: ldr w8, [sp]
  ; CHECK-NEXT: str wzr, [sp]
  ; CHECK-NEXT: str w8, [sp, #8]
  ; CHECK-NEXT: b func
  tail call void @func(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 0, i32 %i)
  ret void
}

declare void @func_i1(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i1 %j)

; CHECK-LABEL: wrapper_func_i1:
define void @wrapper_func_i1(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i1 %j) {
  ; CHECK: // %bb.
  ; CHECK-NEXT: b func_i1
  tail call void @func_i1(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i1 %j)
  ret void
}

declare void @func_signext_i1(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i1 signext %j)

; FIXME: Support zero/sign-extended stack arguments.
; CHECK-LABEL: wrapper_func_i8:
define void @wrapper_func_i8(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i1 signext %j) {
  ; CHECK: // %bb.
  ; CHECK-NEXT: ldrsb w8, [sp, #8]
  ; CHECK-NEXT: strb w8, [sp, #8]
  ; CHECK-NEXT: b func_signext_i1
  tail call void @func_signext_i1(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i1 signext %j)
  ret void
}
