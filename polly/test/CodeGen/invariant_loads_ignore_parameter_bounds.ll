; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting \
; RUN:     -polly-ignore-parameter-bounds -S < %s | FileCheck %s

; CHECK: polly.preload.begin:
; CHECK-NEXT: %global.load = load i32, ptr @global, align 4, !alias.scope !0, !noalias !3
; CHECK-NEXT: store i32 %global.load, ptr %tmp24.preload.s2a

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@global = external global i32

define void @hoge() {
bb:
  %tmp = alloca [4 x double], align 8
  br label %bb18

bb18:                                             ; preds = %bb16
  %tmp19 = load i32, ptr @global, align 4
  br label %bb20

bb20:                                             ; preds = %bb21, %bb18
  %tmp22 = icmp eq i32 0, %tmp19
  br i1 %tmp22, label %bb23, label %bb20

bb23:                                             ; preds = %bb21
  %tmp24 = load i32, ptr @global, align 4
  %tmp25 = add i32 %tmp24, 1
  %tmp26 = sext i32 %tmp25 to i64
  %tmp27 = add nsw i64 %tmp26, -1
  %tmp28 = getelementptr [4 x double], ptr %tmp, i64 0, i64 %tmp27
  store double undef, ptr %tmp28
  br label %bb29

bb29:                                             ; preds = %bb23
  ret void
}
