; RUN: opt -S -loop-reduce < %s | FileCheck %s
;
; Make sure we don't fail an assertion here.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @test() {
; CHECK-LABEL: test
bb:
  br label %bb1

bb1:                                              ; preds = %bb12, %bb
  %tmp2 = phi i64 [ 94, %bb ], [ %tmp20, %bb12 ]
  %tmp3 = phi i32 [ -28407, %bb ], [ %tmp23, %bb12 ]
  %tmp4 = trunc i64 %tmp2 to i32
  %tmp5 = add i32 %tmp3, %tmp4
  %tmp6 = mul i32 undef, %tmp5
  %tmp7 = sub i32 %tmp6, %tmp5
  %tmp8 = shl i32 %tmp7, 1
  %tmp9 = add i32 %tmp8, %tmp3
  %tmp10 = add i32 %tmp9, %tmp4
  %tmp11 = shl i32 %tmp10, 1
  br label %bb21

bb12:                                             ; preds = %bb21
  %tmp13 = mul i32 %tmp22, -101
  %tmp14 = add i32 %tmp22, 2
  %tmp15 = add i32 %tmp14, %tmp13
  %tmp16 = trunc i32 %tmp15 to i8
  %tmp17 = shl i8 %tmp16, 5
  %tmp18 = add i8 %tmp17, 64
  %tmp19 = sext i8 %tmp18 to i32
  %tmp20 = add nsw i64 %tmp2, -3
  br label %bb1

bb21:                                             ; preds = %bb21, %bb1
  %tmp22 = phi i32 [ %tmp11, %bb1 ], [ %tmp23, %bb21 ]
  %tmp23 = add i32 %tmp22, 1
  br i1 false, label %bb12, label %bb21
}
