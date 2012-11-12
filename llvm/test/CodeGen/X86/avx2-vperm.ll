; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

define <8 x i32> @perm_cl_int_8x32(<8 x i32> %A) nounwind readnone {
entry:
; CHECK: perm_cl_int_8x32
; CHECK: vpermd
  %B = shufflevector <8 x i32> %A, <8 x i32> undef, <8 x i32> <i32 0, i32 7, i32 2, i32 1, i32 2, i32 7, i32 6, i32 0>
  ret <8 x i32> %B
}


define <8 x float> @perm_cl_fp_8x32(<8 x float> %A) nounwind readnone {
entry:
; CHECK: perm_cl_fp_8x32
; CHECK: vpermps
  %B = shufflevector <8 x float> %A, <8 x float> undef, <8 x i32> <i32 undef, i32 7, i32 2, i32 undef, i32 4, i32 undef, i32 1, i32 6>
  ret <8 x float> %B
}

define <4 x i64> @perm_cl_int_4x64(<4 x i64> %A) nounwind readnone {
entry:
; CHECK: perm_cl_int_4x64
; CHECK: vpermq
  %B = shufflevector <4 x i64> %A, <4 x i64> undef, <4 x i32> <i32 0, i32 3, i32 2, i32 1>
  ret <4 x i64> %B
}

define <4 x double> @perm_cl_fp_4x64(<4 x double> %A) nounwind readnone {
entry:
; CHECK: perm_cl_fp_4x64
; CHECK: vpermpd
  %B = shufflevector <4 x double> %A, <4 x double> undef, <4 x i32> <i32 0, i32 3, i32 2, i32 1>
  ret <4 x double> %B
}
