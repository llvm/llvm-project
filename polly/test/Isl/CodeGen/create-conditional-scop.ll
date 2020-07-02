; RUN: opt %loadPolly -basic-aa -polly-codegen -verify-loop-info < %s -S | FileCheck %s

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"

@A = common global [1536 x float] zeroinitializer

; This test case used to fail, because we did not add the newly generated basic
; block %polly.start as a basic block to the surrounding loop.
define void @foo() nounwind {
entry:
  br i1 undef, label %while.cond14.preheader, label %for.body7.single_entry.single_entry

while.cond14.preheader:                           ; preds = %for.inc02, %for.body7.single_entry.single_entry, %entry
  ret void

for.body7.single_entry.single_entry:              ; preds = %for.inc02, %entry
  br i1 undef, label %while.cond14.preheader, label %while.body

while.body:                                       ; preds = %while.body, %for.body7.single_entry.single_entry
  %indvar35 = phi i32 [ %0, %while.body ], [ 0, %for.body7.single_entry.single_entry ]
  %ptr = getelementptr [1536 x float], [1536 x float]* @A, i64 0, i32 %indvar35
  store float undef, float* %ptr
  %0 = add i32 %indvar35, 1
  %exitcond2 = icmp eq i32 %0, 42
  br i1 %exitcond2, label %for.inc02, label %while.body

for.inc02:                                        ; preds = %while.body
  br i1 undef, label %while.cond14.preheader, label %for.body7.single_entry.single_entry
}

; CHECK: polly.split_new_and_old
; CHECK: br i1 true, label %polly.start, label %while.body
