; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=core-avx-i | FileCheck %s --check-prefix=AVX

define <2 x double> @v2f2d_ext_vec(<2 x float> %v1) nounwind {
entry:
; CHECK: v2f2d_ext_vec
; CHECK: cvtps2pd
; AVX:   v2f2d_ext_vec
; AVX:   vcvtps2pd
  %f1 = fpext <2 x float> %v1 to <2 x double>
  ret <2 x double> %f1
}

define <3 x double> @v3f2d_ext_vec(<3 x float> %v1) nounwind {
entry:
; CHECK: v3f2d_ext_vec
; CHECK: cvtps2pd
; CHECK: movhlps
; CHECK: cvtps2pd
; AVX:   v3f2d_ext_vec
; AVX:   vcvtps2pd
; AVX:   ret
  %f1 = fpext <3 x float> %v1 to <3 x double>
  ret <3 x double> %f1
}

define <4 x double> @v4f2d_ext_vec(<4 x float> %v1) nounwind {
entry:
; CHECK: v4f2d_ext_vec
; CHECK: cvtps2pd
; CHECK: movhlps
; CHECK: cvtps2pd
; AVX:   v4f2d_ext_vec
; AVX:   vcvtps2pd
; AVX:   ret
  %f1 = fpext <4 x float> %v1 to <4 x double>
  ret <4 x double> %f1
}

define <8 x double> @v8f2d_ext_vec(<8 x float> %v1) nounwind {
entry:
; CHECK: v8f2d_ext_vec
; CHECK: cvtps2pd
; CHECK: cvtps2pd
; CHECK: movhlps
; CHECK: cvtps2pd
; CHECK: movhlps
; CHECK: cvtps2pd
; AVX:   v8f2d_ext_vec
; AVX:   vcvtps2pd
; AVX:   vextractf128
; AVX:   vcvtps2pd
; AVX:   ret
  %f1 = fpext <8 x float> %v1 to <8 x double>
  ret <8 x double> %f1
}

define void @test_vector_creation() nounwind {
  %1 = insertelement <4 x double> undef, double 0.000000e+00, i32 2
  %2 = load double addrspace(1)* null
  %3 = insertelement <4 x double> %1, double %2, i32 3
  store <4 x double> %3, <4 x double>* undef
  ret void
}
