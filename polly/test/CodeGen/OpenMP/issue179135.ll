; RUN: opt  %loadNPMPolly '--passes=polly-custom<delicm;codegen>' --polly-parallel -S < %s | FileCheck %s

; https://github.com/llvm/llvm-project/issues/179135
; CHECK: @func_polly_subfn(

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @func(ptr %arg, i32 %arg1) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb9, %bb
  %i = phi i64 [ 0, %bb ], [ %i10, %bb9 ]
  br label %bb3

bb3:                                              ; preds = %bb3, %bb2
  %i4 = phi i64 [ 0, %bb2 ], [ %i7, %bb3 ]
  %i5 = phi i32 [ 0, %bb2 ], [ %arg1, %bb3 ]
  %i6 = getelementptr i32, ptr %arg, i64 %i
  store i32 %i5, ptr %i6, align 4
  %i7 = add i64 %i4, 1
  %i8 = icmp eq i64 %i4, 1
  br i1 %i8, label %bb9, label %bb3

bb9:                                              ; preds = %bb3
  %i10 = add i64 %i, 1
  %i11 = icmp eq i64 %i, 1
  br i1 %i11, label %bb12, label %bb2

bb12:                                             ; preds = %bb9
  ret void
}
