; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define fastcc void @fix_operands() nounwind {
entry:
  br i1 undef, label %bb3, label %bb1

bb1:                                              ; preds = %bb
  %0 = icmp eq i32 0, undef                       ; <i1> [#uses=1]
  br i1 %0, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1, %bb
  br label %bb14

bb14:                                             ; preds = %bb5, %bb4, %bb3, %entry
  ret void
}

; CHECK: Invalid Scop!
