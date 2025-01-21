; Test the MSA floating-point conversion intrinsics that are encoded with the
; 3RF instruction format.

; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_fexdo_h_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fexdo_h_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fexdo_h_RES  = global <8 x half> <half 0.000000e+00, half 0.000000e+00, half 0.000000e+00, half 0.000000e+00, half 0.000000e+00, half 0.000000e+00, half 0.000000e+00, half 0.000000e+00>, align 16

define void @llvm_mips_fexdo_h_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fexdo_h_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fexdo_h_ARG2
  %2 = tail call <8 x half> @llvm.mips.fexdo.h(<4 x float> %0, <4 x float> %1)
  store <8 x half> %2, ptr @llvm_mips_fexdo_h_RES
  ret void
}

declare <8 x half> @llvm.mips.fexdo.h(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fexdo_h_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fexdo.h
; CHECK: st.h
; CHECK: .size llvm_mips_fexdo_h_test
;
@llvm_mips_fexdo_w_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fexdo_w_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fexdo_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fexdo_w_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fexdo_w_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fexdo_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fexdo.w(<2 x double> %0, <2 x double> %1)
  store <4 x float> %2, ptr @llvm_mips_fexdo_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fexdo.w(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fexdo_w_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fexdo.w
; CHECK: st.w
; CHECK: .size llvm_mips_fexdo_w_test
;
