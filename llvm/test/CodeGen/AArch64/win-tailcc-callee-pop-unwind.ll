; RUN: llc -mtriple=aarch64-windows %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-windows -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s --check-prefix=UNWIND

; A callee-pops function's unwind info describes the pop of its incoming stack
; arguments with a phantom alloc_s before an end_c, so that unwinding out of it
; recovers the caller's SP without the caller needing a frame pointer.

declare void @use(ptr)

; CHECK-LABEL: pop:
; CHECK:         .seh_stackalloc 16
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    str x30, [sp, #-16]!
; CHECK-NEXT:    .seh_save_reg_x x30, 16
; CHECK-NEXT:    .seh_endprologue
define tailcc void @pop(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
  call void @use(ptr null)
  ret void
}

; Nothing to pop, so no phantom opcodes.
; CHECK-LABEL: nopop:
; CHECK-NOT:     .seh_end_c
; CHECK:         .seh_endprologue
define swifttailcc void @nopop(i64 %a) {
  call void @use(ptr null)
  ret void
}

; CHECK-LABEL: leaf:
; CHECK:         .seh_stackalloc 16
; CHECK-NEXT:    .seh_end_c
; CHECK-NEXT:    .seh_endprologue
define swifttailcc void @leaf(i64 %a, i64, i64, i64, i64, i64, i64, i64, i64, i64) {
  ret void
}

; UNWIND-LABEL: Function: pop
; UNWIND:       Prologue [
; UNWIND-NEXT:    {{.*}}; str x30, [sp, #-16]!
; UNWIND-NEXT:    0xe5 ; end_c
; UNWIND-NEXT:    0x01 ; sub sp, #16
; UNWIND-NEXT:    0xe4 ; end
; UNWIND-NEXT:  ]

; UNWIND-LABEL: Function: leaf
; UNWIND:       Prologue [
; UNWIND-NEXT:    0xe5 ; end_c
; UNWIND-NEXT:    0x01 ; sub sp, #16
; UNWIND-NEXT:    0xe4 ; end
; UNWIND-NEXT:  ]
