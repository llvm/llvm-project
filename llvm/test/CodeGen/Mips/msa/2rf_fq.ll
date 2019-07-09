; Test the MSA fixed-point to floating point conversion intrinsics that are
; encoded with the 2RF instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_ffql_w_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ffql_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_ffql_w_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_ffql_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.ffql.w(<8 x i16> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_ffql_w_RES
  ret void
}

declare <4 x float> @llvm.mips.ffql.w(<8 x i16>) nounwind

; CHECK: llvm_mips_ffql_w_test:
; CHECK: ld.h
; CHECK: ffql.w
; CHECK: st.w
; CHECK: .size llvm_mips_ffql_w_test
;
@llvm_mips_ffql_d_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ffql_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_ffql_d_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_ffql_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.ffql.d(<4 x i32> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_ffql_d_RES
  ret void
}

declare <2 x double> @llvm.mips.ffql.d(<4 x i32>) nounwind

; CHECK: llvm_mips_ffql_d_test:
; CHECK: ld.w
; CHECK: ffql.d
; CHECK: st.d
; CHECK: .size llvm_mips_ffql_d_test
;
@llvm_mips_ffqr_w_ARG1 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, align 16
@llvm_mips_ffqr_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_ffqr_w_test() nounwind {
entry:
  %0 = load <8 x i16>, <8 x i16>* @llvm_mips_ffqr_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.ffqr.w(<8 x i16> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_ffqr_w_RES
  ret void
}

declare <4 x float> @llvm.mips.ffqr.w(<8 x i16>) nounwind

; CHECK: llvm_mips_ffqr_w_test:
; CHECK: ld.h
; CHECK: ffqr.w
; CHECK: st.w
; CHECK: .size llvm_mips_ffqr_w_test
;
@llvm_mips_ffqr_d_ARG1 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@llvm_mips_ffqr_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_ffqr_d_test() nounwind {
entry:
  %0 = load <4 x i32>, <4 x i32>* @llvm_mips_ffqr_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.ffqr.d(<4 x i32> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_ffqr_d_RES
  ret void
}

declare <2 x double> @llvm.mips.ffqr.d(<4 x i32>) nounwind

; CHECK: llvm_mips_ffqr_d_test:
; CHECK: ld.w
; CHECK: ffqr.d
; CHECK: st.d
; CHECK: .size llvm_mips_ffqr_d_test
;
