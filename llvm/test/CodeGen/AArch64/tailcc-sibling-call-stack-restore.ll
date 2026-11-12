; RUN: llc -mtriple=aarch64-linux-gnu -O2 < %s | FileCheck %s
; RUN: opt -passes='default<O2>' -S %s \
; RUN:   | llc -mtriple=aarch64-linux-gnu -O2 \
; RUN:   | FileCheck %s --check-prefix=OPT
; RUN: opt -passes='default<O2>' -S %s | FileCheck %s --check-prefix=OPT-IR

declare void @c_callee(ptr)
declare tailcc void @tail_callee(ptr)

; The incoming 16-byte argument area must be restored by the epilogue. A
; sibling call would bypass that restoration.
define tailcc void @tailcc_to_c_with_stack_args(i64 %a, i64 %b, i64 %c, i64 %d,
                                                i64 %e, i64 %f, i64 %g, i64 %h,
                                                i64 %i, i64 %j) {
; CHECK-LABEL: tailcc_to_c_with_stack_args:
; CHECK:       bl c_callee
; CHECK:       add sp, sp, #64
; CHECK:       ret
  %x = alloca [4 x i64]
  store volatile i64 %j, ptr %x
  tail call void @c_callee(ptr null)
  ret void
}

define swifttailcc void @swifttailcc_to_c_with_stack_args(
    i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h,
    i64 %i, i64 %j) {
; CHECK-LABEL: swifttailcc_to_c_with_stack_args:
; CHECK:       bl c_callee
; CHECK:       add sp, sp, #64
; CHECK:       ret
  %x = alloca [4 x i64]
  store volatile i64 %j, ptr %x
  tail call void @c_callee(ptr null)
  ret void
}

; opt adds tail markers to these ordinary calls. They must also lower to
; regular calls so that the epilogue restores the incoming stack arguments.
; OPT-IR-LABEL: define tailcc void @tailcc_to_c_opt_marked(
; OPT-IR: tail call void @c_callee(ptr null)
; OPT-LABEL: tailcc_to_c_opt_marked:
; OPT:       bl c_callee
; OPT:       add sp, sp, #16
; OPT:       ret

define tailcc void @tailcc_to_c_opt_marked(i64 %a, i64 %b, i64 %c, i64 %d,
                                           i64 %e, i64 %f, i64 %g, i64 %h,
                                           i64 %i, i64 %j) {
  %x = alloca [4 x i64]
  store volatile i64 %j, ptr %x
  call void @c_callee(ptr null)
  ret void
}

; OPT-LABEL: swifttailcc_to_c_opt_marked:
; OPT:       bl c_callee
; OPT:       add sp, sp, #16
; OPT:       ret
; OPT-IR-LABEL: define swifttailcc void @swifttailcc_to_c_opt_marked(
; OPT-IR: tail call void @c_callee(ptr null)

define swifttailcc void @swifttailcc_to_c_opt_marked(
    i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h,
    i64 %i, i64 %j) {
  %x = alloca [4 x i64]
  store volatile i64 %j, ptr %x
  call void @c_callee(ptr null)
  ret void
}

; With no incoming stack arguments, the caller has nothing to restore and can
; still use a sibling call.
define tailcc void @tailcc_to_c_without_stack_args() {
; CHECK-LABEL: tailcc_to_c_without_stack_args:
; CHECK:       b c_callee
  tail call void @c_callee(ptr null)
  ret void
}

; Matching guaranteed-tail-call conventions remain eligible even when the
; caller has stack arguments.
define tailcc void @tailcc_to_tailcc_with_stack_args(
    i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h,
    i64 %i, i64 %j) {
; CHECK-LABEL: tailcc_to_tailcc_with_stack_args:
; CHECK:       b tail_callee
  %x = alloca [4 x i64]
  store volatile i64 %j, ptr %x
  tail call tailcc void @tail_callee(ptr null)
  ret void
}
