; RUN: llc -mtriple=aarch64-windows %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-windows -O0 -fast-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-windows -global-isel -global-isel-abort=1 %s -o - | FileCheck %s

; A callee using tailcc/swifttailcc that pops stack arguments leaves SP higher,
; when unwinding back into the caller, than the caller's unwind info expects of
; the argument area it reserved in its prologue. Callers of such functions must
; therefore be found via a frame pointer, which the unwinder uses to recover SP.

declare tailcc void @tail_callee(i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare swifttailcc void @swifttail_callee(i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare tailcc void @tail_callee_regs(i64)
declare swifttailcc void @swifttail_callee_regs(i64)
declare void @c_callee(i64, i64, i64, i64, i64, i64, i64, i64, i64)

; CHECK-LABEL: calls_tailcc:
; CHECK:         .seh_{{(set|add)_fp}}
define void @calls_tailcc() {
  call tailcc void @tail_callee(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

; CHECK-LABEL: calls_swifttailcc:
; CHECK:         .seh_{{(set|add)_fp}}
define void @calls_swifttailcc() {
  call swifttailcc void @swifttail_callee(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

; With no stack arguments there is nothing to pop. The callee's unwind info
; accounts for anything it tail calls.
; CHECK-LABEL: calls_tailcc_regs:
; CHECK-NOT:     .seh_set_fp
; CHECK-NOT:     .seh_add_fp
; CHECK:         .seh_endprologue
define void @calls_tailcc_regs() {
  call tailcc void @tail_callee_regs(i64 1)
  ret void
}

; CHECK-LABEL: calls_swifttailcc_regs:
; CHECK-NOT:     .seh_set_fp
; CHECK-NOT:     .seh_add_fp
; CHECK:         .seh_endprologue
define void @calls_swifttailcc_regs() {
  call swifttailcc void @swifttail_callee_regs(i64 1)
  ret void
}

; Callees using other conventions don't pop their arguments.
; CHECK-LABEL: calls_c:
; CHECK-NOT:     .seh_set_fp
; CHECK-NOT:     .seh_add_fp
; CHECK:         .seh_endprologue
define void @calls_c() {
  call void @c_callee(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

; A tail call doesn't return here, so this function doesn't need one either.
; CHECK-LABEL: tail_calls_swifttailcc:
; CHECK-NOT:     .seh_
; CHECK:         b swifttail_callee_regs
define swifttailcc void @tail_calls_swifttailcc() {
  musttail call swifttailcc void @swifttail_callee_regs(i64 1)
  ret void
}
