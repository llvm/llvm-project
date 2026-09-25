; RUN: opt  %loadNPMPolly '-passes=polly-custom<codegen>' -polly-invariant-load-hoisting -S < %s | FileCheck %s
;
; https://github.com/llvm/llvm-project/issues/190459
; Avoid crash because isl_set_gist_params does not preserve single-valuedness
;
; CHECK: polly.split_new_and_old:

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define i32 @func() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb2, %bb
  %i = phi i32 [ %i3, %bb2 ], [ 0, %bb ]
  br label %bb4

bb2:                                              ; preds = %bb4
  %i3 = add i32 %i, 1
  br label %bb1

bb4:                                              ; preds = %bb4, %bb1
  %i5 = and i32 %i, 7
  %i6 = zext i32 %i5 to i64
  %i7 = getelementptr [8 x i8], ptr null, i64 %i6
  %i8 = load i8, ptr %i7, align 1
  br i1 false, label %bb4, label %bb2
}
