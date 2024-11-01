; RUN: opt -passes=loop-vectorize -force-target-instruction-cost=1 -prefer-predicate-over-epilogue=predicate-dont-vectorize -S < %s 2>&1 | FileCheck %s

; This test currently fails when the LV calculates a maximums safe
; distance for scalable vectors, because the code to eliminate the tail is
; pessimistic when scalable vectors are considered. This will be addressed
; in a future patch, at which point we should be able to un-XFAIL the
; test. The expected output is to vectorize this loop without predication
; (and thus have unpredicated vector store).
; XFAIL: *

; CHECK: store <4 x i32>

target triple = "aarch64"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"


define void @f1(i32* %A) #0 {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %iv
  store i32 1, i32* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, 1024
  br i1 %exitcond, label %for.body, label %exit

exit:
  ret void
}

attributes #0 = { "target-features"="+sve" }
