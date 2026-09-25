; RUN: opt  %loadNPMPolly '-passes=polly-custom<delicm;codegen>' -polly-parallel --polly-parallel-force -S < %s | FileCheck %s

; https://github.com/llvm/llvm-project/issues/185313
; CHECK: @func_polly_subfn(

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @func(i1 %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb2, %bb
  %i = phi i16 [ 0, %bb ], [ %i5, %bb2 ]
  %spec.select = select i1 %arg, i8 0, i8 0
  br i1 %arg, label %bb2, label %bb2

bb2:                                              ; preds = %bb1, %bb1
  %i3 = zext i16 %i to i64
  %i4 = getelementptr i8, ptr null, i64 %i3
  store i8 %spec.select, ptr %i4, align 1
  %i5 = add i16 %i, 1
  %i6 = zext i16 %i to i32
  %i7 = icmp ugt i32 1, %i6
  br i1 %i7, label %bb1, label %bb8

bb8:                                              ; preds = %bb2
  ret void
}
