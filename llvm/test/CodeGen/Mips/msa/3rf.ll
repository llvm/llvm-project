; Test the MSA intrinsics that are encoded with the 3RF instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_fadd_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fadd_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fadd_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fadd_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fadd_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fadd_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @llvm_mips_fadd_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fadd.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fadd_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fadd.w
; CHECK: st.w
; CHECK: .size llvm_mips_fadd_w_test
;
@llvm_mips_fadd_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fadd_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fadd_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fadd_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fadd_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fadd_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @llvm_mips_fadd_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fadd.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fadd_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fadd.d
; CHECK: st.d
; CHECK: .size llvm_mips_fadd_d_test

define void @fadd_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fadd_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fadd_w_ARG2
  %2 = fadd <4 x float> %0, %1
  store <4 x float> %2, <4 x float>* @llvm_mips_fadd_w_RES
  ret void
}

; CHECK: fadd_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fadd.w
; CHECK: st.w
; CHECK: .size fadd_w_test

define void @fadd_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fadd_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fadd_d_ARG2
  %2 = fadd <2 x double> %0, %1
  store <2 x double> %2, <2 x double>* @llvm_mips_fadd_d_RES
  ret void
}

; CHECK: fadd_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fadd.d
; CHECK: st.d
; CHECK: .size fadd_d_test
;
@llvm_mips_fdiv_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fdiv_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fdiv_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fdiv_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fdiv_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fdiv_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fdiv.w(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @llvm_mips_fdiv_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fdiv.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fdiv_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fdiv.w
; CHECK: st.w
; CHECK: .size llvm_mips_fdiv_w_test
;
@llvm_mips_fdiv_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fdiv_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fdiv_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fdiv_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fdiv_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fdiv_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fdiv.d(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @llvm_mips_fdiv_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fdiv.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fdiv_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fdiv.d
; CHECK: st.d
; CHECK: .size llvm_mips_fdiv_d_test

define void @fdiv_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fdiv_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fdiv_w_ARG2
  %2 = fdiv <4 x float> %0, %1
  store <4 x float> %2, <4 x float>* @llvm_mips_fdiv_w_RES
  ret void
}

; CHECK: fdiv_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fdiv.w
; CHECK: st.w
; CHECK: .size fdiv_w_test

define void @fdiv_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fdiv_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fdiv_d_ARG2
  %2 = fdiv <2 x double> %0, %1
  store <2 x double> %2, <2 x double>* @llvm_mips_fdiv_d_RES
  ret void
}

; CHECK: fdiv_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fdiv.d
; CHECK: st.d
; CHECK: .size fdiv_d_test
;
@llvm_mips_fmin_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fmin_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fmin_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fmin_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fmin_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fmin_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fmin.w(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @llvm_mips_fmin_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fmin.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fmin_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fmin.w
; CHECK: st.w
; CHECK: .size llvm_mips_fmin_w_test
;
@llvm_mips_fmin_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fmin_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fmin_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fmin_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fmin_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fmin_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fmin.d(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @llvm_mips_fmin_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fmin.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fmin_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fmin.d
; CHECK: st.d
; CHECK: .size llvm_mips_fmin_d_test
;
@llvm_mips_fmin_a_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fmin_a_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fmin_a_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fmin_a_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fmin_a_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fmin_a_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fmin.a.w(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @llvm_mips_fmin_a_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fmin.a.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fmin_a_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fmin_a.w
; CHECK: st.w
; CHECK: .size llvm_mips_fmin_a_w_test
;
@llvm_mips_fmin_a_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fmin_a_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fmin_a_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fmin_a_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fmin_a_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fmin_a_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fmin.a.d(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @llvm_mips_fmin_a_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fmin.a.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fmin_a_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fmin_a.d
; CHECK: st.d
; CHECK: .size llvm_mips_fmin_a_d_test
;
@llvm_mips_fmax_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fmax_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fmax_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fmax_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fmax_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fmax_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fmax.w(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @llvm_mips_fmax_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fmax.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fmax_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fmax.w
; CHECK: st.w
; CHECK: .size llvm_mips_fmax_w_test
;
@llvm_mips_fmax_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fmax_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fmax_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fmax_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fmax_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fmax_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fmax.d(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @llvm_mips_fmax_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fmax.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fmax_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fmax.d
; CHECK: st.d
; CHECK: .size llvm_mips_fmax_d_test
;
@llvm_mips_fmax_a_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fmax_a_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fmax_a_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fmax_a_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fmax_a_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fmax_a_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fmax.a.w(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @llvm_mips_fmax_a_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fmax.a.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fmax_a_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fmax_a.w
; CHECK: st.w
; CHECK: .size llvm_mips_fmax_a_w_test
;
@llvm_mips_fmax_a_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fmax_a_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fmax_a_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fmax_a_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fmax_a_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fmax_a_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fmax.a.d(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @llvm_mips_fmax_a_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fmax.a.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fmax_a_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fmax_a.d
; CHECK: st.d
; CHECK: .size llvm_mips_fmax_a_d_test
;
@llvm_mips_fmul_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fmul_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fmul_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fmul_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fmul_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fmul_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fmul.w(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @llvm_mips_fmul_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fmul.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fmul_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fmul.w
; CHECK: st.w
; CHECK: .size llvm_mips_fmul_w_test
;
@llvm_mips_fmul_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fmul_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fmul_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fmul_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fmul_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fmul_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fmul.d(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @llvm_mips_fmul_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fmul.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fmul_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fmul.d
; CHECK: st.d
; CHECK: .size llvm_mips_fmul_d_test

define void @fmul_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fmul_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fmul_w_ARG2
  %2 = fmul <4 x float> %0, %1
  store <4 x float> %2, <4 x float>* @llvm_mips_fmul_w_RES
  ret void
}

; CHECK: fmul_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fmul.w
; CHECK: st.w
; CHECK: .size fmul_w_test

define void @fmul_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fmul_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fmul_d_ARG2
  %2 = fmul <2 x double> %0, %1
  store <2 x double> %2, <2 x double>* @llvm_mips_fmul_d_RES
  ret void
}

; CHECK: fmul_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fmul.d
; CHECK: st.d
; CHECK: .size fmul_d_test
;
@llvm_mips_fsub_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsub_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsub_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fsub_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fsub_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fsub_w_ARG2
  %2 = tail call <4 x float> @llvm.mips.fsub.w(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @llvm_mips_fsub_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fsub.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsub_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsub.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsub_w_test
;
@llvm_mips_fsub_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsub_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsub_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fsub_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fsub_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fsub_d_ARG2
  %2 = tail call <2 x double> @llvm.mips.fsub.d(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @llvm_mips_fsub_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fsub.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsub_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsub.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsub_d_test
;

define void @fsub_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fsub_w_ARG1
  %1 = load <4 x float>, <4 x float>* @llvm_mips_fsub_w_ARG2
  %2 = fsub <4 x float> %0, %1
  store <4 x float> %2, <4 x float>* @llvm_mips_fsub_w_RES
  ret void
}

; CHECK: fsub_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsub.w
; CHECK: st.w
; CHECK: .size fsub_w_test

define void @fsub_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fsub_d_ARG1
  %1 = load <2 x double>, <2 x double>* @llvm_mips_fsub_d_ARG2
  %2 = fsub <2 x double> %0, %1
  store <2 x double> %2, <2 x double>* @llvm_mips_fsub_d_RES
  ret void
}

; CHECK: fsub_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsub.d
; CHECK: st.d
; CHECK: .size fsub_d_test
