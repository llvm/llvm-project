; Test the MSA intrinsics that are encoded with the 3RF instruction format and
; take an integer as an operand.

; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_fexp2_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fexp2_w_ARG2 = global <4 x i32> <i32 4, i32 5, i32 6, i32 7>, align 16
@llvm_mips_fexp2_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fexp2_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fexp2_w_ARG1
  %1 = load <4 x i32>, ptr @llvm_mips_fexp2_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fexp2.w(<4 x float> %0, <4 x i32> %1)
  store <4 x float> %2, ptr @llvm_mips_fexp2_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fexp2.w(<4 x float>, <4 x i32>) nounwind

; CHECK: llvm_mips_fexp2_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fexp2.w
; CHECK: st.w
; CHECK: .size llvm_mips_fexp2_w_test
;
@llvm_mips_fexp2_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fexp2_d_ARG2 = global <2 x i64> <i64 2, i64 3>, align 16
@llvm_mips_fexp2_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fexp2_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fexp2_d_ARG1
  %1 = load <2 x i64>, ptr @llvm_mips_fexp2_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fexp2.d(<2 x double> %0, <2 x i64> %1)
  store <2 x double> %2, ptr @llvm_mips_fexp2_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fexp2.d(<2 x double>, <2 x i64>) nounwind

; CHECK: llvm_mips_fexp2_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fexp2.d
; CHECK: st.d
; CHECK: .size llvm_mips_fexp2_d_test
;
