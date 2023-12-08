; Test the MSA floating point conversion intrinsics (e.g. float->double) that
; are encoded with the 2RF instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_fexupl_w_ARG1 = global <8 x half> <half 0.000000e+00, half 1.000000e+00, half 2.000000e+00, half 3.000000e+00, half 4.000000e+00, half 5.000000e+00, half 6.000000e+00, half 7.000000e+00>, align 16
@llvm_mips_fexupl_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fexupl_w_test() nounwind {
entry:
  %0 = load <8 x half>, ptr @llvm_mips_fexupl_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.fexupl.w(<8 x half> %0)
  store <4 x float> %1, ptr @llvm_mips_fexupl_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fexupl.w(<8 x half>) nounwind

; CHECK: llvm_mips_fexupl_w_test:
; CHECK: ld.h
; CHECK: fexupl.w
; CHECK: st.w
; CHECK: .size llvm_mips_fexupl_w_test
;
@llvm_mips_fexupl_d_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fexupl_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fexupl_d_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fexupl_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.fexupl.d(<4 x float> %0)
  store <2 x double> %1, ptr @llvm_mips_fexupl_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fexupl.d(<4 x float>) nounwind

; CHECK: llvm_mips_fexupl_d_test:
; CHECK: ld.w
; CHECK: fexupl.d
; CHECK: st.d
; CHECK: .size llvm_mips_fexupl_d_test
;
@llvm_mips_fexupr_w_ARG1 = global <8 x half> <half 0.000000e+00, half 1.000000e+00, half 2.000000e+00, half 3.000000e+00, half 4.000000e+00, half 5.000000e+00, half 6.000000e+00, half 7.000000e+00>, align 16
@llvm_mips_fexupr_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fexupr_w_test() nounwind {
entry:
  %0 = load <8 x half>, ptr @llvm_mips_fexupr_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.fexupr.w(<8 x half> %0)
  store <4 x float> %1, ptr @llvm_mips_fexupr_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fexupr.w(<8 x half>) nounwind

; CHECK: llvm_mips_fexupr_w_test:
; CHECK: ld.h
; CHECK: fexupr.w
; CHECK: st.w
; CHECK: .size llvm_mips_fexupr_w_test
;
@llvm_mips_fexupr_d_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fexupr_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fexupr_d_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fexupr_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.fexupr.d(<4 x float> %0)
  store <2 x double> %1, ptr @llvm_mips_fexupr_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fexupr.d(<4 x float>) nounwind

; CHECK: llvm_mips_fexupr_d_test:
; CHECK: ld.w
; CHECK: fexupr.d
; CHECK: st.d
; CHECK: .size llvm_mips_fexupr_d_test
;
