; RUN: llc -mtriple=mipsel -mattr=mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=CHECK-STATIC16

@x = common global float 0.000000e+00, align 4

define void @foo() nounwind {
entry:
  %0 = load float, float* @x, align 4
  %1 = load float, float* @x, align 4
  %mul = fmul float %0, %1
  store float %mul, float* @x, align 4
; CHECK-STATIC16: jal	__mips16_mulsf3
  ret void
}
