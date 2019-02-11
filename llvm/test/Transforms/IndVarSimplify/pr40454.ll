; RUN: opt -S -indvars  < %s | FileCheck %s
; REQUIRES: asserts
; XFAIL: *

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @test() {
; CHECK-LABEL: @test

bb:
  br label %bb2

bb1:                                              ; No predecessors!
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %tmp = phi i32 [ -9, %bb ], [ %tmp6, %bb1 ]
  br label %bb3

bb3:                                              ; preds = %bb10, %bb2
  %tmp4 = phi i32 [ -9, %bb2 ], [ %tmp6, %bb10 ]
  br i1 undef, label %bb5, label %bb12

bb5:                                              ; preds = %bb3
  %tmp6 = add i32 %tmp4, -1
  %tmp7 = zext i32 %tmp6 to i64
  br i1 undef, label %bb8, label %bb9

bb8:                                              ; preds = %bb5
  br label %bb10

bb9:                                              ; preds = %bb5
  br label %bb10

bb10:                                             ; preds = %bb9, %bb8
  %tmp11 = and i64 undef, %tmp7
  br label %bb3

bb12:                                             ; preds = %bb3
  ret void
}
