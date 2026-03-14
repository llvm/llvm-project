; RUN: not llc < %s -mtriple=i686-pc-linux -o - -mattr=+sse2 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc < %s -mtriple=x86_64-pc-linux -o - -mattr=+mmx | FileCheck %s

; ERR: error: couldn't allocate input reg for constraint 'x'
define void @foo(fp128 %x) {
; CHECK-LABEL: foo:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    movaps {{\.?LCPI[0-9]+_[0-9]+}}(%rip), %xmm1
; CHECK-NEXT:    callq __multf3@PLT
; CHECK-NEXT:    #APP
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    retq
entry:
  %mul = fmul fp128 %x, 0xL00000000000000003FFF800000000000
  tail call void asm sideeffect "", "x,~{dirflag},~{fpsr},~{flags}"(fp128 %mul)
  ret void
}
