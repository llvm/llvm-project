; Test the MSA intrinsics that are encoded with the 3RF instruction format and
; use the result as a third operand.

; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_fmadd_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fmadd_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fmadd_w_ARG3 = global <4 x float> <float 8.000000e+00, float 9.000000e+00, float 1.000000e+01, float 1.100000e+01>, align 16
@llvm_mips_fmadd_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fmadd_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fmadd_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fmadd_w_ARG2
  %2 = load <4 x float>, ptr @llvm_mips_fmadd_w_ARG3
  %3 = tail call <4 x float> @llvm.mips.fmadd.w(<4 x float> %0, <4 x float> %1, <4 x float> %2)
  store <4 x float> %3, ptr @llvm_mips_fmadd_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fmadd.w(<4 x float>, <4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fmadd_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fmadd.w
; CHECK: st.w
; CHECK: .size llvm_mips_fmadd_w_test
;
@llvm_mips_fmadd_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fmadd_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fmadd_d_ARG3 = global <2 x double> <double 4.000000e+00, double 5.000000e+00>, align 16
@llvm_mips_fmadd_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fmadd_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fmadd_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fmadd_d_ARG2
  %2 = load <2 x double>, ptr @llvm_mips_fmadd_d_ARG3
  %3 = tail call <2 x double> @llvm.mips.fmadd.d(<2 x double> %0, <2 x double> %1, <2 x double> %2)
  store <2 x double> %3, ptr @llvm_mips_fmadd_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fmadd.d(<2 x double>, <2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fmadd_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fmadd.d
; CHECK: st.d
; CHECK: .size llvm_mips_fmadd_d_test
;
@llvm_mips_fmsub_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fmsub_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fmsub_w_ARG3 = global <4 x float> <float 8.000000e+00, float 9.000000e+00, float 1.000000e+01, float 1.100000e+01>, align 16
@llvm_mips_fmsub_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fmsub_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fmsub_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fmsub_w_ARG2
  %2 = load <4 x float>, ptr @llvm_mips_fmsub_w_ARG3
  %3 = tail call <4 x float> @llvm.mips.fmsub.w(<4 x float> %0, <4 x float> %1, <4 x float> %2)
  store <4 x float> %3, ptr @llvm_mips_fmsub_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fmsub.w(<4 x float>, <4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fmsub_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fmsub.w
; CHECK: st.w
; CHECK: .size llvm_mips_fmsub_w_test
;
@llvm_mips_fmsub_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fmsub_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fmsub_d_ARG3 = global <2 x double> <double 4.000000e+00, double 5.000000e+00>, align 16
@llvm_mips_fmsub_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fmsub_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fmsub_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fmsub_d_ARG2
  %2 = load <2 x double>, ptr @llvm_mips_fmsub_d_ARG3
  %3 = tail call <2 x double> @llvm.mips.fmsub.d(<2 x double> %0, <2 x double> %1, <2 x double> %2)
  store <2 x double> %3, ptr @llvm_mips_fmsub_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fmsub.d(<2 x double>, <2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fmsub_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fmsub.d
; CHECK: st.d
; CHECK: .size llvm_mips_fmsub_d_test
;
define <4 x float> @llvm_mips_fmadd_w_const_test() nounwind {
; CHECK-LABEL: llvm_mips_fmadd_w_const_test:
; CHECK:       lui $1, 16608
  %r = tail call <4 x float> @llvm.mips.fmadd.w(<4 x float> splat (float 1.0), <4 x float> splat (float 2.0), <4 x float> splat (float 3.0))
  ret <4 x float> %r
}

define <2 x double> @llvm_mips_fmadd_d_const_test() nounwind {
; The const-evaluated value is stored in the constant pool.
; CHECK:       .8byte 0x401c000000000000 # double 7
; CHECK-LABEL: llvm_mips_fmadd_d_const_test:
  %r = tail call <2 x double> @llvm.mips.fmadd.d(<2 x double> splat (double 1.0), <2 x double> splat (double 2.0), <2 x double> splat (double 3.0))
  ret <2 x double> %r
}
