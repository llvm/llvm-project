; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -disable-fp-elim -asm-verbose=0 | FileCheck %s

define void @f() {
  ret void
}

; CHECK:       .cfi_startproc
; CHECK-NEXT:  pushq
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  .cfi_offset %rbp, -16
; CHECK-NEXT:  movq    %rsp, %rbp
; CHECK-NEXT:  .cfi_def_cfa_register %rbp
; CHECK-NEXT:  popq    %rbp
; CHECK-NEXT:  .cfi_def_cfa %rsp, 8
; CHECK-NEXT:  ret
