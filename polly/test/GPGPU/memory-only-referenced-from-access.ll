; RUN: opt %loadPolly -polly-codegen-ppcg -polly-acc-dump-kernel-ir \
; RUN: -polly-invariant-load-hoisting -polly-ignore-aliasing \
; RUN: -polly-process-unprofitable -polly-ignore-parameter-bounds \
; RUN: -polly-acc-fail-on-verify-module-failure \
; RUN: -polly-acc-codegen-managed-memory \
; RUN: -disable-output < %s | \
; RUN: FileCheck %s

; REQUIRES: pollyacc

; Verify that we correctly generate a kernel even if certain invariant load
; hoisted parameters appear only in memory accesses, but not domain elements.

; CHECK: @FUNC_quux_SCOP_0_KERNEL_0(ptr addrspace(1) %MemRef_tmp4, i32 %tmp3, i32 %tmp, i32 %tmp31, i32 %tmp2)

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct.hoge = type { ptr, i64, i64, [1 x %struct.widget] }
%struct.widget = type { i64, i64, i64 }

@global = external unnamed_addr global %struct.hoge, align 32

define void @quux(ptr noalias %arg, ptr noalias %arg1) {
bb:
  %tmp = load i32, ptr %arg, align 4
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = load i32, ptr %arg1, align 4
  %tmp4 = load ptr, ptr @global, align 32
  br label %bb5

bb5:                                              ; preds = %bb5, %bb
  %tmp6 = phi i32 [ %tmp11, %bb5 ], [ 0, %bb ]
  %tmp7 = sext i32 %tmp6 to i64
  %tmp8 = sub nsw i64 %tmp7, %tmp2
  %tmp9 = getelementptr [0 x double], ptr %tmp4, i64 0, i64 %tmp8
  store double undef, ptr %tmp9, align 8
  %tmp10 = icmp eq i32 %tmp6, %tmp3
  %tmp11 = add i32 %tmp6, 1
  br i1 %tmp10, label %bb12, label %bb5

bb12:                                             ; preds = %bb5
  ret void
}
