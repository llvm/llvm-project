; RUN: opt  %loadNPMPolly '-passes=polly-custom<codegen>' -polly-invariant-load-hoisting -S < %s | FileCheck %s

; https://github.com/llvm/llvm-project/issues/190459
; Avoid a crash due to comparing integers with different widths

; "i1 false" is the result of the comparison with the fixed width
; CHECK: br i1 false, label %polly.preload.exec, label %polly.preload.merge

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define i32 @func(i32 %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  store i16 0, ptr null, align 2
  %i = icmp eq i32 %arg, 0
  br i1 %i, label %bb2, label %bb5

bb2:                                              ; preds = %bb1
  %i3 = load ptr, ptr null, align 8
  %i4 = load i16, ptr %i3, align 2
  br label %bb5

bb5:                                              ; preds = %bb2, %bb1
  ret i32 0
}
