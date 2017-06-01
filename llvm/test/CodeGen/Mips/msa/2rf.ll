; Test the MSA intrinsics that are encoded with the 2RF instruction format.

; RUN: llc -march=mips -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 -relocation-model=pic < %s | FileCheck %s

@llvm_mips_flog2_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_flog2_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_flog2_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_flog2_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.flog2.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_flog2_w_RES
  ret void
}

declare <4 x float> @llvm.mips.flog2.w(<4 x float>) nounwind

; CHECK: llvm_mips_flog2_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_flog2_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: flog2.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_flog2_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_flog2_w_test
;
@llvm_mips_flog2_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_flog2_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_flog2_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_flog2_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.flog2.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_flog2_d_RES
  ret void
}

declare <2 x double> @llvm.mips.flog2.d(<2 x double>) nounwind

; CHECK: llvm_mips_flog2_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_flog2_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: flog2.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_flog2_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_flog2_d_test

define void @flog2_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_flog2_w_ARG1
  %1 = tail call <4 x float> @llvm.log2.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_flog2_w_RES
  ret void
}

declare <4 x float> @llvm.log2.v4f32(<4 x float> %val)

; CHECK: flog2_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_flog2_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: flog2.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_flog2_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size flog2_w_test

define void @flog2_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_flog2_d_ARG1
  %1 = tail call <2 x double> @llvm.log2.v2f64(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_flog2_d_RES
  ret void
}

declare <2 x double> @llvm.log2.v2f64(<2 x double> %val)

; CHECK: flog2_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_flog2_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: flog2.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_flog2_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size flog2_d_test
;
@llvm_mips_frint_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_frint_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_frint_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_frint_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.frint.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_frint_w_RES
  ret void
}

declare <4 x float> @llvm.mips.frint.w(<4 x float>) nounwind

; CHECK: llvm_mips_frint_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_frint_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: frint.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_frint_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_frint_w_test
;
@llvm_mips_frint_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_frint_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_frint_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_frint_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.frint.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_frint_d_RES
  ret void
}

declare <2 x double> @llvm.mips.frint.d(<2 x double>) nounwind

; CHECK: llvm_mips_frint_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_frint_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: frint.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_frint_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_frint_d_test

define void @frint_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_frint_w_ARG1
  %1 = tail call <4 x float> @llvm.rint.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_frint_w_RES
  ret void
}

declare <4 x float> @llvm.rint.v4f32(<4 x float>) nounwind

; CHECK: frint_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_frint_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: frint.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_frint_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size frint_w_test

define void @frint_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_frint_d_ARG1
  %1 = tail call <2 x double> @llvm.rint.v2f64(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_frint_d_RES
  ret void
}

declare <2 x double> @llvm.rint.v2f64(<2 x double>) nounwind

; CHECK: frint_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_frint_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: frint.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_frint_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size frint_d_test
;
@llvm_mips_frcp_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_frcp_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_frcp_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_frcp_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.frcp.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_frcp_w_RES
  ret void
}

declare <4 x float> @llvm.mips.frcp.w(<4 x float>) nounwind

; CHECK: llvm_mips_frcp_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_frcp_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: frcp.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_frcp_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_frcp_w_test
;
@llvm_mips_frcp_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_frcp_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_frcp_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_frcp_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.frcp.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_frcp_d_RES
  ret void
}

declare <2 x double> @llvm.mips.frcp.d(<2 x double>) nounwind

; CHECK: llvm_mips_frcp_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_frcp_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: frcp.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_frcp_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_frcp_d_test
;
@llvm_mips_frsqrt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_frsqrt_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_frsqrt_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_frsqrt_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.frsqrt.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_frsqrt_w_RES
  ret void
}

declare <4 x float> @llvm.mips.frsqrt.w(<4 x float>) nounwind

; CHECK: llvm_mips_frsqrt_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_frsqrt_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: frsqrt.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_frsqrt_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_frsqrt_w_test
;
@llvm_mips_frsqrt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_frsqrt_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_frsqrt_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_frsqrt_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.frsqrt.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_frsqrt_d_RES
  ret void
}

declare <2 x double> @llvm.mips.frsqrt.d(<2 x double>) nounwind

; CHECK: llvm_mips_frsqrt_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_frsqrt_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: frsqrt.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_frsqrt_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_frsqrt_d_test
;
@llvm_mips_fsqrt_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fsqrt_w_RES  = global <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, align 16

define void @llvm_mips_fsqrt_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fsqrt_w_ARG1
  %1 = tail call <4 x float> @llvm.mips.fsqrt.w(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_fsqrt_w_RES
  ret void
}

declare <4 x float> @llvm.mips.fsqrt.w(<4 x float>) nounwind

; CHECK: llvm_mips_fsqrt_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_fsqrt_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: fsqrt.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_fsqrt_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_fsqrt_w_test
;
@llvm_mips_fsqrt_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fsqrt_d_RES  = global <2 x double> <double 0.000000e+00, double 0.000000e+00>, align 16

define void @llvm_mips_fsqrt_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fsqrt_d_ARG1
  %1 = tail call <2 x double> @llvm.mips.fsqrt.d(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_fsqrt_d_RES
  ret void
}

declare <2 x double> @llvm.mips.fsqrt.d(<2 x double>) nounwind

; CHECK: llvm_mips_fsqrt_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_fsqrt_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: fsqrt.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_fsqrt_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_fsqrt_d_test

define void @fsqrt_w_test() nounwind {
entry:
  %0 = load <4 x float>, <4 x float>* @llvm_mips_fsqrt_w_ARG1
  %1 = tail call <4 x float> @llvm.sqrt.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* @llvm_mips_fsqrt_w_RES
  ret void
}

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) nounwind

; CHECK: fsqrt_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_fsqrt_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: fsqrt.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_fsqrt_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size fsqrt_w_test

define void @fsqrt_d_test() nounwind {
entry:
  %0 = load <2 x double>, <2 x double>* @llvm_mips_fsqrt_d_ARG1
  %1 = tail call <2 x double> @llvm.sqrt.v2f64(<2 x double> %0)
  store <2 x double> %1, <2 x double>* @llvm_mips_fsqrt_d_RES
  ret void
}

declare <2 x double> @llvm.sqrt.v2f64(<2 x double>) nounwind

; CHECK: fsqrt_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_fsqrt_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: fsqrt.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_fsqrt_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size fsqrt_d_test
;
