; RUN: opt < %s -passes=indvars -S -scalar-evolution-use-expensive-range-sharpening | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @test() {
; CHECK-LABEL: test

bb:
  br label %bb1

bb1:                                              ; preds = %bb10, %bb
  %tmp = phi i32 [ undef, %bb ], [ %tmp11, %bb10 ]
  %tmp2 = phi i32 [ 0, %bb ], [ 1, %bb10 ]
  br i1 false, label %bb3, label %bb4

bb3:                                              ; preds = %bb1
  br label %bb8

bb4:                                              ; preds = %bb1
  br label %bb16

bb5:                                              ; preds = %bb16
  %tmp6 = phi i64 [ %tmp21, %bb16 ]
  %tmp7 = phi i64 [ undef, %bb16 ]
  br label %bb8

bb8:                                              ; preds = %bb5, %bb3
  %tmp9 = phi i64 [ undef, %bb3 ], [ %tmp6, %bb5 ]
  br label %bb13

bb10:                                             ; preds = %bb13
  %tmp11 = phi i32 [ %tmp15, %bb13 ]
  br i1 undef, label %bb12, label %bb1

bb12:                                             ; preds = %bb10
  ret void

bb13:                                             ; preds = %bb13, %bb8
  %tmp14 = phi i32 [ %tmp, %bb8 ], [ %tmp15, %bb13 ]
  %tmp15 = add i32 %tmp14, undef
  br i1 undef, label %bb10, label %bb13

bb16:                                             ; preds = %bb16, %bb4
  %tmp17 = phi i32 [ %tmp27, %bb16 ], [ %tmp2, %bb4 ]
  %tmp18 = phi i64 [ %tmp21, %bb16 ], [ undef, %bb4 ]
  %tmp19 = sext i32 %tmp17 to i64
  %tmp20 = mul i64 undef, %tmp19
  %tmp21 = add i64 %tmp18, 1
  %tmp22 = add i32 %tmp17, %tmp
  %tmp23 = add i32 %tmp22, undef
  %tmp24 = add i32 %tmp23, undef
  %tmp25 = and i32 %tmp24, 31
  %tmp26 = lshr i32 undef, %tmp25
  %tmp27 = add nsw i32 %tmp17, 1
  %tmp28 = icmp sgt i32 %tmp17, 111
  br i1 %tmp28, label %bb5, label %bb16
}
