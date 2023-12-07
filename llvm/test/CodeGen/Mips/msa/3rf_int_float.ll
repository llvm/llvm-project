; Test the MSA intrinsics that are encoded with the 3RF instruction format and
; produce an integer as a result.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@llvm_mips_fcaf_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcaf_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcaf_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcaf_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcaf_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcaf_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcaf.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcaf_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcaf.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcaf_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcaf.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcaf_w_test
;
@llvm_mips_fcaf_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcaf_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcaf_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcaf_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcaf_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcaf_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcaf.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcaf_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcaf.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcaf_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcaf.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcaf_d_test
;
@llvm_mips_fceq_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fceq_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fceq_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fceq_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fceq_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fceq_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fceq.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fceq_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fceq.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fceq_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fceq.w
; CHECK: st.w
; CHECK: .size llvm_mips_fceq_w_test
;
@llvm_mips_fceq_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fceq_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fceq_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fceq_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fceq_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fceq_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fceq.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fceq_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fceq.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fceq_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fceq.d
; CHECK: st.d
; CHECK: .size llvm_mips_fceq_d_test
;
@llvm_mips_fcle_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcle_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcle_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcle_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcle_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcle_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcle.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcle_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcle.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcle_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcle.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcle_w_test
;
@llvm_mips_fcle_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcle_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcle_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcle_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcle_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcle_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcle.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcle_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcle.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcle_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcle.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcle_d_test
;
@llvm_mips_fclt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fclt_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fclt_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fclt_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fclt_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fclt_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fclt.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fclt_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fclt.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fclt_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fclt.w
; CHECK: st.w
; CHECK: .size llvm_mips_fclt_w_test
;
@llvm_mips_fclt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fclt_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fclt_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fclt_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fclt_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fclt_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fclt.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fclt_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fclt.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fclt_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fclt.d
; CHECK: st.d
; CHECK: .size llvm_mips_fclt_d_test
;
@llvm_mips_fcor_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcor_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcor_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcor_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcor_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcor_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcor.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcor_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcor.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcor_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcor.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcor_w_test
;
@llvm_mips_fcor_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcor_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcor_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcor_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcor_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcor_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcor.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcor_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcor.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcor_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcor.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcor_d_test
;
@llvm_mips_fcne_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcne_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcne_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcne_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcne_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcne_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcne.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcne_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcne.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcne_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcne.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcne_w_test
;
@llvm_mips_fcne_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcne_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcne_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcne_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcne_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcne_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcne.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcne_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcne.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcne_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcne.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcne_d_test
;
@llvm_mips_fcueq_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcueq_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcueq_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcueq_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcueq_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcueq_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcueq.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcueq_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcueq.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcueq_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcueq.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcueq_w_test
;
@llvm_mips_fcueq_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcueq_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcueq_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcueq_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcueq_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcueq_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcueq.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcueq_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcueq.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcueq_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcueq.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcueq_d_test
;
@llvm_mips_fcult_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcult_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcult_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcult_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcult_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcult_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcult.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcult_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcult.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcult_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcult.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcult_w_test
;
@llvm_mips_fcult_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcult_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcult_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcult_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcult_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcult_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcult.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcult_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcult.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcult_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcult.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcult_d_test
;
@llvm_mips_fcule_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcule_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcule_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcule_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcule_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcule_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcule.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcule_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcule.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcule_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcule.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcule_w_test
;
@llvm_mips_fcule_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcule_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcule_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcule_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcule_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcule_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcule.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcule_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcule.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcule_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcule.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcule_d_test
;
@llvm_mips_fcun_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcun_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcun_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcun_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcun_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcun_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcun.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcun_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcun.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcun_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcun.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcun_w_test
;
@llvm_mips_fcun_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcun_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcun_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcun_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcun_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcun_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcun.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcun_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcun.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcun_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcun.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcun_d_test
;
@llvm_mips_fcune_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fcune_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fcune_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fcune_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fcune_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fcune_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fcune.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fcune_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fcune.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fcune_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fcune.w
; CHECK: st.w
; CHECK: .size llvm_mips_fcune_w_test
;
@llvm_mips_fcune_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fcune_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fcune_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fcune_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fcune_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fcune_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fcune.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fcune_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fcune.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fcune_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fcune.d
; CHECK: st.d
; CHECK: .size llvm_mips_fcune_d_test
;
@llvm_mips_fsaf_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsaf_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsaf_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsaf_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsaf_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsaf_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsaf.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsaf_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsaf.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsaf_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsaf.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsaf_w_test
;
@llvm_mips_fsaf_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsaf_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsaf_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsaf_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsaf_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsaf_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsaf.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsaf_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsaf.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsaf_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsaf.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsaf_d_test
;
@llvm_mips_fseq_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fseq_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fseq_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fseq_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fseq_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fseq_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fseq.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fseq_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fseq.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fseq_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fseq.w
; CHECK: st.w
; CHECK: .size llvm_mips_fseq_w_test
;
@llvm_mips_fseq_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fseq_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fseq_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fseq_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fseq_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fseq_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fseq.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fseq_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fseq.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fseq_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fseq.d
; CHECK: st.d
; CHECK: .size llvm_mips_fseq_d_test
;
@llvm_mips_fsle_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsle_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsle_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsle_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsle_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsle_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsle.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsle_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsle.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsle_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsle.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsle_w_test
;
@llvm_mips_fsle_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsle_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsle_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsle_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsle_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsle_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsle.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsle_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsle.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsle_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsle.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsle_d_test
;
@llvm_mips_fslt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fslt_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fslt_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fslt_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fslt_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fslt_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fslt.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fslt_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fslt.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fslt_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fslt.w
; CHECK: st.w
; CHECK: .size llvm_mips_fslt_w_test
;
@llvm_mips_fslt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fslt_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fslt_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fslt_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fslt_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fslt_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fslt.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fslt_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fslt.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fslt_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fslt.d
; CHECK: st.d
; CHECK: .size llvm_mips_fslt_d_test
;
@llvm_mips_fsor_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsor_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsor_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsor_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsor_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsor_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsor.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsor_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsor.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsor_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsor.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsor_w_test
;
@llvm_mips_fsor_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsor_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsor_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsor_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsor_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsor_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsor.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsor_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsor.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsor_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsor.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsor_d_test
;
@llvm_mips_fsne_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsne_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsne_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsne_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsne_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsne_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsne.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsne_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsne.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsne_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsne.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsne_w_test
;
@llvm_mips_fsne_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsne_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsne_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsne_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsne_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsne_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsne.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsne_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsne.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsne_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsne.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsne_d_test
;
@llvm_mips_fsueq_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsueq_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsueq_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsueq_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsueq_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsueq_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsueq.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsueq_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsueq.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsueq_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsueq.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsueq_w_test
;
@llvm_mips_fsueq_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsueq_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsueq_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsueq_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsueq_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsueq_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsueq.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsueq_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsueq.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsueq_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsueq.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsueq_d_test
;
@llvm_mips_fsult_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsult_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsult_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsult_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsult_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsult_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsult.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsult_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsult.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsult_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsult.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsult_w_test
;
@llvm_mips_fsult_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsult_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsult_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsult_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsult_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsult_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsult.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsult_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsult.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsult_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsult.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsult_d_test
;
@llvm_mips_fsule_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsule_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsule_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsule_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsule_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsule_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsule.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsule_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsule.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsule_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsule.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsule_w_test
;
@llvm_mips_fsule_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsule_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsule_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsule_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsule_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsule_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsule.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsule_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsule.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsule_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsule.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsule_d_test
;
@llvm_mips_fsun_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsun_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsun_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsun_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsun_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsun_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsun.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsun_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsun.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsun_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsun.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsun_w_test
;
@llvm_mips_fsun_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsun_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsun_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsun_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsun_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsun_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsun.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsun_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsun.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsun_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsun.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsun_d_test
;
@llvm_mips_fsune_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsune_w_ARG2 = global <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, align 16
@llvm_mips_fsune_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fsune_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fsune_w_ARG1
  %1 = load <4 x float>, ptr @llvm_mips_fsune_w_ARG2
  %2 = tail call <4 x i32> @llvm.mips.fsune.w(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, ptr @llvm_mips_fsune_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fsune.w(<4 x float>, <4 x float>) nounwind

; CHECK: llvm_mips_fsune_w_test:
; CHECK: ld.w
; CHECK: ld.w
; CHECK: fsune.w
; CHECK: st.w
; CHECK: .size llvm_mips_fsune_w_test
;
@llvm_mips_fsune_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsune_d_ARG2 = global <2 x double> <double 2.000000e+00, double 3.000000e+00>, align 16
@llvm_mips_fsune_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fsune_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fsune_d_ARG1
  %1 = load <2 x double>, ptr @llvm_mips_fsune_d_ARG2
  %2 = tail call <2 x i64> @llvm.mips.fsune.d(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, ptr @llvm_mips_fsune_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fsune.d(<2 x double>, <2 x double>) nounwind

; CHECK: llvm_mips_fsune_d_test:
; CHECK: ld.d
; CHECK: ld.d
; CHECK: fsune.d
; CHECK: st.d
; CHECK: .size llvm_mips_fsune_d_test
;
