; RUN: llc -mtriple x86_64-w64-windows-gnu -filetype=asm -exception-model=dwarf -o - %s | FileCheck %s

define void @_Z1fv() {
entry:
  tail call void asm sideeffect "", "~{xmm10},~{xmm15},~{dirflag},~{fpsr},~{flags}"()
  ret void
}

; CHECK-LABEL: _Z1fv:
; CHECK:   .cfi_startproc
; CHECK:   subq    $40, %rsp
; CHECK:   movaps  %xmm15, 16(%rsp)
; CHECK:   movaps  %xmm10, (%rsp)
; CHECK:   .cfi_def_cfa_offset 48
; CHECK:   .cfi_offset %xmm10, -48
; CHECK:   .cfi_offset %xmm15, -32
; CHECK:   movaps  (%rsp), %xmm10
; CHECK:   movaps  16(%rsp), %xmm15
; CHECK:   addq    $40, %rsp
; CHECK:   retq
; CHECK:   .cfi_endproc
