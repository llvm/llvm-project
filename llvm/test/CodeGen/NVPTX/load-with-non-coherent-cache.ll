; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck -check-prefix=SM20 %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck -check-prefix=SM35 %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify -arch=sm_35 %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; SM20-LABEL: .visible .entry foo1(
; SM20: ld.global.f32
; SM35-LABEL: .visible .entry foo1(
; SM35: ld.global.nc.f32
define void @foo1(float * noalias readonly %from, float * %to) {
  %1 = load float, float * %from
  store float %1, float * %to
  ret void
}

; SM20-LABEL: .visible .entry foo2(
; SM20: ld.global.f64
; SM35-LABEL: .visible .entry foo2(
; SM35: ld.global.nc.f64
define void @foo2(double * noalias readonly %from, double * %to) {
  %1 = load double, double * %from
  store double %1, double * %to
  ret void
}

; SM20-LABEL: .visible .entry foo3(
; SM20: ld.global.u16
; SM35-LABEL: .visible .entry foo3(
; SM35: ld.global.nc.u16
define void @foo3(i16 * noalias readonly %from, i16 * %to) {
  %1 = load i16, i16 * %from
  store i16 %1, i16 * %to
  ret void
}

; SM20-LABEL: .visible .entry foo4(
; SM20: ld.global.u32
; SM35-LABEL: .visible .entry foo4(
; SM35: ld.global.nc.u32
define void @foo4(i32 * noalias readonly %from, i32 * %to) {
  %1 = load i32, i32 * %from
  store i32 %1, i32 * %to
  ret void
}

; SM20-LABEL: .visible .entry foo5(
; SM20: ld.global.u64
; SM35-LABEL: .visible .entry foo5(
; SM35: ld.global.nc.u64
define void @foo5(i64 * noalias readonly %from, i64 * %to) {
  %1 = load i64, i64 * %from
  store i64 %1, i64 * %to
  ret void
}

; i128 is non standard integer in nvptx64
; SM20-LABEL: .visible .entry foo6(
; SM20: ld.global.u64
; SM20: ld.global.u64
; SM35-LABEL: .visible .entry foo6(
; SM35: ld.global.nc.u64
; SM35: ld.global.nc.u64
define void @foo6(i128 * noalias readonly %from, i128 * %to) {
  %1 = load i128, i128 * %from
  store i128 %1, i128 * %to
  ret void
}

; SM20-LABEL: .visible .entry foo7(
; SM20: ld.global.v2.u8
; SM35-LABEL: .visible .entry foo7(
; SM35: ld.global.nc.v2.u8
define void @foo7(<2 x i8> * noalias readonly %from, <2 x i8> * %to) {
  %1 = load <2 x i8>, <2 x i8> * %from
  store <2 x i8> %1, <2 x i8> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo8(
; SM20: ld.global.v2.u16
; SM35-LABEL: .visible .entry foo8(
; SM35: ld.global.nc.v2.u16
define void @foo8(<2 x i16> * noalias readonly %from, <2 x i16> * %to) {
  %1 = load <2 x i16>, <2 x i16> * %from
  store <2 x i16> %1, <2 x i16> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo9(
; SM20: ld.global.v2.u32
; SM35-LABEL: .visible .entry foo9(
; SM35: ld.global.nc.v2.u32
define void @foo9(<2 x i32> * noalias readonly %from, <2 x i32> * %to) {
  %1 = load <2 x i32>, <2 x i32> * %from
  store <2 x i32> %1, <2 x i32> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo10(
; SM20: ld.global.v2.u64
; SM35-LABEL: .visible .entry foo10(
; SM35: ld.global.nc.v2.u64
define void @foo10(<2 x i64> * noalias readonly %from, <2 x i64> * %to) {
  %1 = load <2 x i64>, <2 x i64> * %from
  store <2 x i64> %1, <2 x i64> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo11(
; SM20: ld.global.v2.f32
; SM35-LABEL: .visible .entry foo11(
; SM35: ld.global.nc.v2.f32
define void @foo11(<2 x float> * noalias readonly %from, <2 x float> * %to) {
  %1 = load <2 x float>, <2 x float> * %from
  store <2 x float> %1, <2 x float> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo12(
; SM20: ld.global.v2.f64
; SM35-LABEL: .visible .entry foo12(
; SM35: ld.global.nc.v2.f64
define void @foo12(<2 x double> * noalias readonly %from, <2 x double> * %to) {
  %1 = load <2 x double>, <2 x double> * %from
  store <2 x double> %1, <2 x double> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo13(
; SM20: ld.global.v4.u8
; SM35-LABEL: .visible .entry foo13(
; SM35: ld.global.nc.v4.u8
define void @foo13(<4 x i8> * noalias readonly %from, <4 x i8> * %to) {
  %1 = load <4 x i8>, <4 x i8> * %from
  store <4 x i8> %1, <4 x i8> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo14(
; SM20: ld.global.v4.u16
; SM35-LABEL: .visible .entry foo14(
; SM35: ld.global.nc.v4.u16
define void @foo14(<4 x i16> * noalias readonly %from, <4 x i16> * %to) {
  %1 = load <4 x i16>, <4 x i16> * %from
  store <4 x i16> %1, <4 x i16> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo15(
; SM20: ld.global.v4.u32
; SM35-LABEL: .visible .entry foo15(
; SM35: ld.global.nc.v4.u32
define void @foo15(<4 x i32> * noalias readonly %from, <4 x i32> * %to) {
  %1 = load <4 x i32>, <4 x i32> * %from
  store <4 x i32> %1, <4 x i32> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo16(
; SM20: ld.global.v4.f32
; SM35-LABEL: .visible .entry foo16(
; SM35: ld.global.nc.v4.f32
define void @foo16(<4 x float> * noalias readonly %from, <4 x float> * %to) {
  %1 = load <4 x float>, <4 x float> * %from
  store <4 x float> %1, <4 x float> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo17(
; SM20: ld.global.v2.f64
; SM20: ld.global.v2.f64
; SM35-LABEL: .visible .entry foo17(
; SM35: ld.global.nc.v2.f64
; SM35: ld.global.nc.v2.f64
define void @foo17(<4 x double> * noalias readonly %from, <4 x double> * %to) {
  %1 = load <4 x double>, <4 x double> * %from
  store <4 x double> %1, <4 x double> * %to
  ret void
}

; SM20-LABEL: .visible .entry foo18(
; SM20: ld.global.u64
; SM35-LABEL: .visible .entry foo18(
; SM35: ld.global.nc.u64
define void @foo18(float ** noalias readonly %from, float ** %to) {
  %1 = load float *, float ** %from
  store float * %1, float ** %to
  ret void
}

; Test that we can infer a cached load for a pointer induction variable.
; SM20-LABEL: .visible .entry foo19(
; SM20: ld.global.f32
; SM35-LABEL: .visible .entry foo19(
; SM35: ld.global.nc.f32
define void @foo19(float * noalias readonly %from, float * %to, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %nexti, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %nextsum, %loop ]
  %ptr = getelementptr inbounds float, float * %from, i32 %i
  %value = load float, float * %ptr, align 4
  %nextsum = fadd float %value, %sum
  %nexti = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  store float %nextsum, float * %to
  ret void
}

; This test captures the case of a non-kernel function. In a
; non-kernel function, without interprocedural analysis, we do not
; know that the parameter is global. We also do not know that the
; pointed-to memory is never written to (for the duration of the
; kernel). For both reasons, we cannot use a cached load here.
; SM20-LABEL: notkernel(
; SM20: ld.f32
; SM35-LABEL: notkernel(
; SM35: ld.f32
define void @notkernel(float * noalias readonly %from, float * %to) {
  %1 = load float, float * %from
  store float %1, float * %to
  ret void
}

; As @notkernel, but with the parameter explicitly marked as global. We still
; do not know that the parameter is never written to (for the duration of the
; kernel). This case does not currently come up normally since we do not infer
; that pointers are global interprocedurally as of 2015-08-05.
; SM20-LABEL: notkernel2(
; SM20: ld.global.f32
; SM35-LABEL: notkernel2(
; SM35: ld.global.f32
define void @notkernel2(float addrspace(1) * noalias readonly %from, float * %to) {
  %1 = load float, float addrspace(1) * %from
  store float %1, float * %to
  ret void
}

!nvvm.annotations = !{!1 ,!2 ,!3 ,!4 ,!5 ,!6, !7 ,!8 ,!9 ,!10 ,!11 ,!12, !13, !14, !15, !16, !17, !18, !19}
!1 = !{void (float *, float *)* @foo1, !"kernel", i32 1}
!2 = !{void (double *, double *)* @foo2, !"kernel", i32 1}
!3 = !{void (i16 *, i16 *)* @foo3, !"kernel", i32 1}
!4 = !{void (i32 *, i32 *)* @foo4, !"kernel", i32 1}
!5 = !{void (i64 *, i64 *)* @foo5, !"kernel", i32 1}
!6 = !{void (i128 *, i128 *)* @foo6, !"kernel", i32 1}
!7 = !{void (<2 x i8> *, <2 x i8> *)* @foo7, !"kernel", i32 1}
!8 = !{void (<2 x i16> *, <2 x i16> *)* @foo8, !"kernel", i32 1}
!9 = !{void (<2 x i32> *, <2 x i32> *)* @foo9, !"kernel", i32 1}
!10 = !{void (<2 x i64> *, <2 x i64> *)* @foo10, !"kernel", i32 1}
!11 = !{void (<2 x float> *, <2 x float> *)* @foo11, !"kernel", i32 1}
!12 = !{void (<2 x double> *, <2 x double> *)* @foo12, !"kernel", i32 1}
!13 = !{void (<4 x i8> *, <4 x i8> *)* @foo13, !"kernel", i32 1}
!14 = !{void (<4 x i16> *, <4 x i16> *)* @foo14, !"kernel", i32 1}
!15 = !{void (<4 x i32> *, <4 x i32> *)* @foo15, !"kernel", i32 1}
!16 = !{void (<4 x float> *, <4 x float> *)* @foo16, !"kernel", i32 1}
!17 = !{void (<4 x double> *, <4 x double> *)* @foo17, !"kernel", i32 1}
!18 = !{void (float **, float **)* @foo18, !"kernel", i32 1}
!19 = !{void (float *, float *, i32)* @foo19, !"kernel", i32 1}
