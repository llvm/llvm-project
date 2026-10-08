; RUN: llc < %s -mtriple=aarch64-windows | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-windows -O0 -fast-isel | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-windows -global-isel -global-isel-abort=1 | FileCheck %s

; A callee using tailcc/swifttailcc may tail call something that needs more
; stack argument space, which leaves SP lower than the caller's unwind info
; expects when unwinding back into the caller. Callers must therefore be found
; via a frame pointer, which the unwinder uses to recover SP.

declare tailcc void @tail_callee()
declare swifttailcc void @swifttail_callee()
declare void @c_callee()

; CHECK-LABEL: calls_tailcc:
; CHECK:         .seh_{{(set|add)_fp}}
define void @calls_tailcc() {
  call tailcc void @tail_callee()
  ret void
}

; CHECK-LABEL: calls_swifttailcc:
; CHECK:         .seh_{{(set|add)_fp}}
define void @calls_swifttailcc() {
  call swifttailcc void @swifttail_callee()
  ret void
}

; Calls using other conventions don't need one.
; CHECK-LABEL: calls_c:
; CHECK-NOT:     .seh_set_fp
; CHECK-NOT:     .seh_add_fp
; CHECK:         .seh_endprologue
define void @calls_c() {
  call void @c_callee()
  ret void
}

; A tail call doesn't return here, so this function doesn't need one either.
; CHECK-LABEL: tail_calls_swifttailcc:
; CHECK-NOT:     .seh_
; CHECK:         b swifttail_callee
define swifttailcc void @tail_calls_swifttailcc() {
  musttail call swifttailcc void @swifttail_callee()
  ret void
}
