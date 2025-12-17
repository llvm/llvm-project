; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+sse2 < %s | FileCheck %s

define i32 @scalar_fptosi_fold(float %x) {
; CHECK-LABEL: scalar_fptosi_fold:
; CHECK-NOT: movsd
; CHECK-NOT: andpd
; CHECK: cvttss2si
  %abs = call float @llvm.fabs.f32(float %x)
  %cmp = fcmp olt float %abs, 2147483648.0
  %conv = fptosi float %x to i32
  %res = select i1 %cmp, i32 %conv, i32 -2147483648
  ret i32 %res
}

define <4 x i32> @vector_fptosi_fold(<4 x float> %x) {
; CHECK-LABEL: vector_fptosi_fold:
; CHECK-NOT: cmpltps
; CHECK-NOT: andps
; CHECK: cvttps2dq
  %abs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %x)
  %cmp = fcmp olt <4 x float> %abs,
                  <float 2147483648.0, float 2147483648.0,
                   float 2147483648.0, float 2147483648.0>
  %conv = fptosi <4 x float> %x to <4 x i32>
  %res = select <4 x i1> %cmp, <4 x i32> %conv,
                     <4 x i32> <i32 -2147483648, i32 -2147483648,
                                i32 -2147483648, i32 -2147483648>
  ret <4 x i32> %res
}

declare float @llvm.fabs.f32(float)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
