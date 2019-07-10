; Test the MSA floating-point to fixed-point conversion intrinsics that are
; encoded with the 2RF instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_ftq_h_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftq_h_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_ftq_h_RES  = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, align 16

define void @llvm_mips_ftq_h_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_ftq_h_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_ftq_h_ARG2
  %2 = tail call <8 x i16> @llvm.mips.ftq.h(<4 x float> %0, <4 x float> %1)
  store <8 x i16> %2, <8 x i16>* @llvm_mips_ftq_h_RES
  ret void
}

declare <8 x i16> @llvm.mips.ftq.h(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_ftq_h_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ftq.h
; CHECK: st.h
; CHECK: .size llvm_mips_ftq_h_test
;
@llvm_mips_ftq_w_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftq_w_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_ftq_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftq_w_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_ftq_w_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_ftq_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.ftq.w(<2 x double> %0, <2 x double> %1)
  store <4 x i32> %2, <4 x i32>* @llvm_mips_ftq_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftq.w(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_ftq_w_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: ftq.w
; CHECK: st.w
; CHECK: .size llvm_mips_ftq_w_test
;
