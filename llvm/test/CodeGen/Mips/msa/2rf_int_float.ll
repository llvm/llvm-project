; Test the MSA floating point to integer intrinsics that are encoded with the
; 2RF instruction format. This includes conversions but other instructions such
; as fclass are also here.

; RUN: llc -mtriple=mips-elf -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-elf -mattr=+msa,+fp64,+mips32r2 -relocation-model=pic < %s | FileCheck %s

@llvm_mips_fclass_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_fclass_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_fclass_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_fclass_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.fclass.w(<4 x float> %0)
  store <4 x i32> %1, ptr @llvm_mips_fclass_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.fclass.w(<4 x float>) nounwind

; CHECK: llvm_mips_fclass_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_fclass_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: fclass.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_fclass_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_fclass_w_test
;
@llvm_mips_fclass_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_fclass_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_fclass_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_fclass_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.fclass.d(<2 x double> %0)
  store <2 x i64> %1, ptr @llvm_mips_fclass_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.fclass.d(<2 x double>) nounwind

; CHECK: llvm_mips_fclass_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_fclass_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: fclass.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_fclass_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_fclass_d_test
;
@llvm_mips_ftrunc_s_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftrunc_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftrunc_s_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_ftrunc_s_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ftrunc.s.w(<4 x float> %0)
  store <4 x i32> %1, ptr @llvm_mips_ftrunc_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftrunc.s.w(<4 x float>) nounwind

; CHECK: llvm_mips_ftrunc_s_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_ftrunc_s_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ftrunc_s.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_ftrunc_s_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_ftrunc_s_w_test
;
@llvm_mips_ftrunc_s_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftrunc_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ftrunc_s_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_ftrunc_s_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ftrunc.s.d(<2 x double> %0)
  store <2 x i64> %1, ptr @llvm_mips_ftrunc_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ftrunc.s.d(<2 x double>) nounwind

; CHECK: llvm_mips_ftrunc_s_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_ftrunc_s_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ftrunc_s.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_ftrunc_s_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_ftrunc_s_d_test
;
@llvm_mips_ftrunc_u_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftrunc_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftrunc_u_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_ftrunc_u_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ftrunc.u.w(<4 x float> %0)
  store <4 x i32> %1, ptr @llvm_mips_ftrunc_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftrunc.u.w(<4 x float>) nounwind

; CHECK: llvm_mips_ftrunc_u_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_ftrunc_u_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ftrunc_u.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_ftrunc_u_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_ftrunc_u_w_test
;
@llvm_mips_ftrunc_u_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftrunc_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ftrunc_u_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_ftrunc_u_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ftrunc.u.d(<2 x double> %0)
  store <2 x i64> %1, ptr @llvm_mips_ftrunc_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ftrunc.u.d(<2 x double>) nounwind

; CHECK: llvm_mips_ftrunc_u_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_ftrunc_u_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ftrunc_u.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_ftrunc_u_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_ftrunc_u_d_test
;
@llvm_mips_ftint_s_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftint_s_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftint_s_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_ftint_s_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ftint.s.w(<4 x float> %0)
  store <4 x i32> %1, ptr @llvm_mips_ftint_s_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftint.s.w(<4 x float>) nounwind

; CHECK: llvm_mips_ftint_s_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_ftint_s_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ftint_s.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_ftint_s_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_ftint_s_w_test
;
@llvm_mips_ftint_s_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftint_s_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ftint_s_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_ftint_s_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ftint.s.d(<2 x double> %0)
  store <2 x i64> %1, ptr @llvm_mips_ftint_s_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ftint.s.d(<2 x double>) nounwind

; CHECK: llvm_mips_ftint_s_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_ftint_s_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ftint_s.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_ftint_s_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_ftint_s_d_test
;
@llvm_mips_ftint_u_w_ARG1 = global <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, align 16
@llvm_mips_ftint_u_w_RES  = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 16

define void @llvm_mips_ftint_u_w_test() nounwind {
entry:
  %0 = load <4 x float>, ptr @llvm_mips_ftint_u_w_ARG1
  %1 = tail call <4 x i32> @llvm.mips.ftint.u.w(<4 x float> %0)
  store <4 x i32> %1, ptr @llvm_mips_ftint_u_w_RES
  ret void
}

declare <4 x i32> @llvm.mips.ftint.u.w(<4 x float>) nounwind

; CHECK: llvm_mips_ftint_u_w_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_ftint_u_w_ARG1)
; CHECK-DAG: ld.w [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ftint_u.w [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_ftint_u_w_RES)
; CHECK-DAG: st.w [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_ftint_u_w_test
;
@llvm_mips_ftint_u_d_ARG1 = global <2 x double> <double 0.000000e+00, double 1.000000e+00>, align 16
@llvm_mips_ftint_u_d_RES  = global <2 x i64> <i64 0, i64 0>, align 16

define void @llvm_mips_ftint_u_d_test() nounwind {
entry:
  %0 = load <2 x double>, ptr @llvm_mips_ftint_u_d_ARG1
  %1 = tail call <2 x i64> @llvm.mips.ftint.u.d(<2 x double> %0)
  store <2 x i64> %1, ptr @llvm_mips_ftint_u_d_RES
  ret void
}

declare <2 x i64> @llvm.mips.ftint.u.d(<2 x double>) nounwind

; CHECK: llvm_mips_ftint_u_d_test:
; CHECK-DAG: lw [[R1:\$[0-9]+]], %got(llvm_mips_ftint_u_d_ARG1)
; CHECK-DAG: ld.d [[WS:\$w[0-9]+]], 0([[R1]])
; CHECK-DAG: ftint_u.d [[WD:\$w[0-9]+]], [[WS]]
; CHECK-DAG: lw [[R2:\$[0-9]+]], %got(llvm_mips_ftint_u_d_RES)
; CHECK-DAG: st.d [[WD]], 0([[R2]])
; CHECK: .size llvm_mips_ftint_u_d_test
;
