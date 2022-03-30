; RUN: llc -mtriple x86_64-w64-windows-gnu -filetype=asm -exception-model=dwarf -o - %s | FileCheck %s
; RUN: llc < %s -mtriple x86_64-w64-windows-gnu -exception-model=dwarf -stop-after=prologepilog | FileCheck %s --check-prefix=PEI

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

; PEI-LABEL: name: _Z1fv
; PEI:         $rsp = frame-setup SUB64ri8 $rsp, 40, implicit-def dead $eflags
; PEI-NEXT:    frame-setup MOVAPSmr $rsp, 1, $noreg, 16, $noreg, killed $xmm15 :: (store (s128) into %fixed-stack.1)
; PEI-NEXT:    frame-setup MOVAPSmr $rsp, 1, $noreg, 0, $noreg, killed $xmm10 :: (store (s128) into %fixed-stack.0)
; PEI-NEXT:    {{^ +}}CFI_INSTRUCTION def_cfa_offset 48
; PEI-NEXT:    {{^ +}}CFI_INSTRUCTION offset $xmm10, -48
; PEI-NEXT:    {{^ +}}CFI_INSTRUCTION offset $xmm15, -32
; PEI-NEXT:    INLINEASM {{.*}}
; PEI-NEXT:    $xmm10 = MOVAPSrm $rsp, 1, $noreg, 0, $noreg :: (load (s128) from %fixed-stack.0)
; PEI-NEXT:    $xmm15 = MOVAPSrm $rsp, 1, $noreg, 16, $noreg :: (load (s128) from %fixed-stack.1)
; PEI-NEXT:    $rsp = frame-destroy ADD64ri8 $rsp, 40, implicit-def dead $eflags
; PEI-NEXT:    RET 0
