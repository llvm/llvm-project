; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define <4 x float> @foo(<4 x float> %val, <4 x float> %test) nounwind {
; CHECK-LABEL: LCPI0_0:
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-LABEL: foo:
; CHECK: cmpeqps %xmm1, %xmm0
; CHECK-NEXT: andps LCPI0_0(%rip), %xmm0
; CHECK-NEXT: retq

  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %result = sitofp <4 x i32> %ext to <4 x float>
  ret <4 x float> %result
}

; Make sure the operation doesn't try to get folded when the sizes don't match,
; as that ends up crashing later when trying to form a bitcast operation for
; the folded nodes.
define void @foo1(<4 x float> %val, <4 x float> %test, <4 x double>* %p) nounwind {
; CHECK-LABEL: LCPI1_0:
; CHECK-NEXT: .long 1                       ## 0x1
; CHECK-NEXT: .long 1                       ## 0x1
; CHECK-NEXT: .long 1                       ## 0x1
; CHECK-NEXT: .long 1                       ## 0x1
; CHECK-LABEL: foo1:
;   FIXME: The operation gets scalarized. If/when the compiler learns to better
;          use [V]CVTDQ2PD, this will need updated.
; CHECK: cvtsi2sdq
; CHECK: cvtsi2sdq
; CHECK: cvtsi2sdq
; CHECK: cvtsi2sdq
  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %result = sitofp <4 x i32> %ext to <4 x double>
  store <4 x double> %result, <4 x double>* %p
  ret void
}

; Also test the general purpose constant folding of int->fp.
define void @foo2(<4 x float>* noalias %result) nounwind {
; CHECK-LABEL: LCPI2_0:
; CHECK-NEXT: .long 1082130432              ## float 4.000000e+00
; CHECK-NEXT: .long 1084227584              ## float 5.000000e+00
; CHECK-NEXT: .long 1086324736              ## float 6.000000e+00
; CHECK-NEXT: .long 1088421888              ## float 7.000000e+00
; CHECK-LABEL: foo2:
; CHECK:  movaps LCPI2_0(%rip), %xmm0

  %val = uitofp <4 x i32> <i32 4, i32 5, i32 6, i32 7> to <4 x float>
  store <4 x float> %val, <4 x float>* %result
  ret void
}
