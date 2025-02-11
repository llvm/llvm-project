; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck -check-prefix=SM20 %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 -verify-machineinstrs | FileCheck -check-prefix=SM35 %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; SM20-LABEL: .visible .entry foo1(
; SM20: ld.global.f32
; SM35-LABEL: .visible .entry foo1(
; SM35: ld.global.nc.f32
define ptx_kernel void @foo1(ptr noalias readonly %from, ptr %to) {
  %1 = load float, ptr %from
  store float %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo2(
; SM20: ld.global.f64
; SM35-LABEL: .visible .entry foo2(
; SM35: ld.global.nc.f64
define ptx_kernel void @foo2(ptr noalias readonly %from, ptr %to) {
  %1 = load double, ptr %from
  store double %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo3(
; SM20: ld.global.u16
; SM35-LABEL: .visible .entry foo3(
; SM35: ld.global.nc.u16
define ptx_kernel void @foo3(ptr noalias readonly %from, ptr %to) {
  %1 = load i16, ptr %from
  store i16 %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo4(
; SM20: ld.global.u32
; SM35-LABEL: .visible .entry foo4(
; SM35: ld.global.nc.u32
define ptx_kernel void @foo4(ptr noalias readonly %from, ptr %to) {
  %1 = load i32, ptr %from
  store i32 %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo5(
; SM20: ld.global.u64
; SM35-LABEL: .visible .entry foo5(
; SM35: ld.global.nc.u64
define ptx_kernel void @foo5(ptr noalias readonly %from, ptr %to) {
  %1 = load i64, ptr %from
  store i64 %1, ptr %to
  ret void
}

; i128 is non standard integer in nvptx64
; SM20-LABEL: .visible .entry foo6(
; SM20: ld.global.u64
; SM20: ld.global.u64
; SM35-LABEL: .visible .entry foo6(
; SM35: ld.global.nc.u64
; SM35: ld.global.nc.u64
define ptx_kernel void @foo6(ptr noalias readonly %from, ptr %to) {
  %1 = load i128, ptr %from
  store i128 %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo7(
; SM20: ld.global.v2.u8
; SM35-LABEL: .visible .entry foo7(
; SM35: ld.global.nc.v2.u8
define ptx_kernel void @foo7(ptr noalias readonly %from, ptr %to) {
  %1 = load <2 x i8>, ptr %from
  store <2 x i8> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo8(
; SM20: ld.global.u32
; SM35-LABEL: .visible .entry foo8(
; SM35: ld.global.nc.u32
define ptx_kernel void @foo8(ptr noalias readonly %from, ptr %to) {
  %1 = load <2 x i16>, ptr %from
  store <2 x i16> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo9(
; SM20: ld.global.v2.u32
; SM35-LABEL: .visible .entry foo9(
; SM35: ld.global.nc.v2.u32
define ptx_kernel void @foo9(ptr noalias readonly %from, ptr %to) {
  %1 = load <2 x i32>, ptr %from
  store <2 x i32> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo10(
; SM20: ld.global.v2.u64
; SM35-LABEL: .visible .entry foo10(
; SM35: ld.global.nc.v2.u64
define ptx_kernel void @foo10(ptr noalias readonly %from, ptr %to) {
  %1 = load <2 x i64>, ptr %from
  store <2 x i64> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo11(
; SM20: ld.global.v2.f32
; SM35-LABEL: .visible .entry foo11(
; SM35: ld.global.nc.v2.f32
define ptx_kernel void @foo11(ptr noalias readonly %from, ptr %to) {
  %1 = load <2 x float>, ptr %from
  store <2 x float> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo12(
; SM20: ld.global.v2.f64
; SM35-LABEL: .visible .entry foo12(
; SM35: ld.global.nc.v2.f64
define ptx_kernel void @foo12(ptr noalias readonly %from, ptr %to) {
  %1 = load <2 x double>, ptr %from
  store <2 x double> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo13(
; SM20: ld.global.u32
; SM35-LABEL: .visible .entry foo13(
; SM35: ld.global.nc.u32
define ptx_kernel void @foo13(ptr noalias readonly %from, ptr %to) {
  %1 = load <4 x i8>, ptr %from
  store <4 x i8> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo14(
; SM20: ld.global.v4.u16
; SM35-LABEL: .visible .entry foo14(
; SM35: ld.global.nc.v4.u16
define ptx_kernel void @foo14(ptr noalias readonly %from, ptr %to) {
  %1 = load <4 x i16>, ptr %from
  store <4 x i16> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo15(
; SM20: ld.global.v4.u32
; SM35-LABEL: .visible .entry foo15(
; SM35: ld.global.nc.v4.u32
define ptx_kernel void @foo15(ptr noalias readonly %from, ptr %to) {
  %1 = load <4 x i32>, ptr %from
  store <4 x i32> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo16(
; SM20: ld.global.v4.f32
; SM35-LABEL: .visible .entry foo16(
; SM35: ld.global.nc.v4.f32
define ptx_kernel void @foo16(ptr noalias readonly %from, ptr %to) {
  %1 = load <4 x float>, ptr %from
  store <4 x float> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo17(
; SM20: ld.global.v2.f64
; SM20: ld.global.v2.f64
; SM35-LABEL: .visible .entry foo17(
; SM35: ld.global.nc.v2.f64
; SM35: ld.global.nc.v2.f64
define ptx_kernel void @foo17(ptr noalias readonly %from, ptr %to) {
  %1 = load <4 x double>, ptr %from
  store <4 x double> %1, ptr %to
  ret void
}

; SM20-LABEL: .visible .entry foo18(
; SM20: ld.global.u64
; SM35-LABEL: .visible .entry foo18(
; SM35: ld.global.nc.u64
define ptx_kernel void @foo18(ptr noalias readonly %from, ptr %to) {
  %1 = load ptr, ptr %from
  store ptr %1, ptr %to
  ret void
}

; Test that we can infer a cached load for a pointer induction variable.
; SM20-LABEL: .visible .entry foo19(
; SM20: ld.global.f32
; SM35-LABEL: .visible .entry foo19(
; SM35: ld.global.nc.f32
define ptx_kernel void @foo19(ptr noalias readonly %from, ptr %to, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %nexti, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %nextsum, %loop ]
  %ptr = getelementptr inbounds float, ptr %from, i32 %i
  %value = load float, ptr %ptr, align 4
  %nextsum = fadd float %value, %sum
  %nexti = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  store float %nextsum, ptr %to
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
define void @notkernel(ptr noalias readonly %from, ptr %to) {
  %1 = load float, ptr %from
  store float %1, ptr %to
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
define void @notkernel2(ptr addrspace(1) noalias readonly %from, ptr %to) {
  %1 = load float, ptr addrspace(1) %from
  store float %1, ptr %to
  ret void
}
