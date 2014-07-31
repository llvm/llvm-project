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

define void @bar(<4 x float>* noalias %result) nounwind {
; CHECK-LABEL: LCPI1_0:
; CHECK-NEXT: .long 1082130432              ## float 4.000000e+00
; CHECK-NEXT: .long 1084227584              ## float 5.000000e+00
; CHECK-NEXT: .long 1086324736              ## float 6.000000e+00
; CHECK-NEXT: .long 1088421888              ## float 7.000000e+00
; CHECK-LABEL: bar:
; CHECK:  movaps LCPI1_0(%rip), %xmm0

  %val = uitofp <4 x i32> <i32 4, i32 5, i32 6, i32 7> to <4 x float>
  store <4 x float> %val, <4 x float>* %result
  ret void
}
