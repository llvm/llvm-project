; RUN: llc < %s -mtriple=i386-linux -mattr=+avx | FileCheck %s

; In i386 there are only 8 XMMs (xmm0-xmm7), make sure we are not creating illegal XMM
define float @only_xmm0_7(i32 %arg) {
top:
  tail call void asm sideeffect "", "~{xmm0},~{xmm1},~{xmm2},~{xmm3},~{dirflag},~{fpsr},~{flags}"()
  tail call void asm sideeffect "", "~{xmm4},~{xmm5},~{xmm6},~{xmm7},~{dirflag},~{fpsr},~{flags}"()
  %tmp1 = sitofp i32 %arg to float
  ret float %tmp1
;CHECK-LABEL:@only_xmm0_7
;CHECK: vcvtsi2ssl {{.*}}, {{%xmm[0-7]+}}, {{%xmm[0-7]+}}
}
