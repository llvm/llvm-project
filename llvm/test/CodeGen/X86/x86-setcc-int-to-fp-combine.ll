; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define <4 x float> @foo(<4 x float> %val, <4 x float> %test) nounwind {
; CHECK-LABEL: LCPI0_0:
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
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
; CHECK: cvtdq2pd
; CHECK: cvtdq2pd
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

; Fold explicit AND operations when the constant isn't a splat of a single
; scalar value like what the zext creates.
define <4 x float> @foo3(<4 x float> %val, <4 x float> %test) nounwind {
; CHECK-LABEL: LCPI3_0:
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 0                       ## 0x0
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 0                       ## 0x0
; CHECK-LABEL: foo3:
; CHECK: cmpeqps %xmm1, %xmm0
; CHECK-NEXT: andps LCPI3_0(%rip), %xmm0
  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %and = and <4 x i32> %ext, <i32 255, i32 256, i32 257, i32 258>
  %result = sitofp <4 x i32> %and to <4 x float>
  ret <4 x float> %result
}

; Test the general purpose constant folding of uint->fp.
define void @foo4(<4 x float>* noalias %result) nounwind {
; CHECK-LABEL: LCPI4_0:
; CHECK-NEXT: .long 1065353216              ## float 1.000000e+00
; CHECK-NEXT: .long 1123942400              ## float 1.270000e+02
; CHECK-NEXT: .long 1124073472              ## float 1.280000e+02
; CHECK-NEXT: .long 1132396544              ## float 2.550000e+02
; CHECK-LABEL: foo4:
; CHECK:  movaps LCPI4_0(%rip), %xmm0

  %val = uitofp <4 x i8> <i8 1, i8 127, i8 -128, i8 -1> to <4 x float>
  store <4 x float> %val, <4 x float>* %result
  ret void
}
