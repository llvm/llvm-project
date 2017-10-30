; RUN: opt %loadPolly -S -polly-process-unprofitable -polly-codegen-ppcg \
; RUN: -polly-invariant-load-hoisting -polly-ignore-parameter-bounds < %s | \
; RUN: FileCheck %s

; REQUIRES: pollyacc

; CHECK: polly_launchKernel

; Verify that this program compiles. At some point, this compilation crashed
; due to insufficient parameters being available.

source_filename = "bugpoint-output-4d01492.bc"
target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct.barney = type { i8*, i64, i64, [2 x %struct.widget] }
%struct.widget = type { i64, i64, i64 }

@global = external unnamed_addr global %struct.barney, align 32

; Function Attrs: nounwind uwtable
define void @wobble(i32* noalias %arg) #0 {
bb:
  %tmp = load i32, i32* %arg, align 4
  br label %bb1

bb1:                                              ; preds = %bb13, %bb
  %tmp2 = phi i32 [ %tmp15, %bb13 ], [ 1, %bb ]
  br label %bb3

bb3:                                              ; preds = %bb3, %bb1
  %tmp4 = load i32*, i32** bitcast (%struct.barney* @global to i32**), align 32
  %tmp5 = sext i32 %tmp2 to i64
  %tmp6 = load i64, i64* getelementptr inbounds (%struct.barney, %struct.barney* @global, i64 0, i32 3, i64 1, i32 0), align 8
  %tmp7 = mul i64 %tmp6, %tmp5
  %tmp8 = add i64 %tmp7, 0
  %tmp9 = load i64, i64* getelementptr inbounds (%struct.barney, %struct.barney* @global, i64 0, i32 1), align 8
  %tmp10 = add i64 %tmp8, %tmp9
  %tmp11 = getelementptr i32, i32* %tmp4, i64 %tmp10
  store i32 undef, i32* %tmp11, align 4
  %tmp12 = icmp eq i32 0, 0
  br i1 %tmp12, label %bb13, label %bb3

bb13:                                             ; preds = %bb3
  %tmp14 = icmp eq i32 %tmp2, %tmp
  %tmp15 = add i32 %tmp2, 1
  br i1 %tmp14, label %bb16, label %bb1

bb16:                                             ; preds = %bb13
  ret void
}

attributes #0 = { nounwind uwtable }
