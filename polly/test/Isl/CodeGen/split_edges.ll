; RUN: opt %loadPolly -polly-codegen-isl -verify-region-info -verify-dom-info -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-pc-linux-gnu"

define void @loop_with_condition() nounwind {
bb0:
  fence seq_cst
  br label %bb1

bb1:
  br i1 true, label %bb2, label %bb3

bb2:
  %ind1 = phi i32 [0, %bb1], [ %inc1, %bb2]
  %inc1 = add i32 %ind1, 1
  %cond1 = icmp eq i32 %ind1, 32
  br i1 %cond1, label %bb4, label %bb2

bb3:
  %ind2 = phi i32 [0, %bb1], [ %inc2, %bb3]
  %inc2 = add i32 %ind2, 1
  br i1 true, label %bb4, label %bb3

bb4:
  br label %bb5

bb5:
  fence seq_cst
  ret void

}

; CHECK: polly.split_new_and_old
; CHECK: polly.merge_new_and_old
