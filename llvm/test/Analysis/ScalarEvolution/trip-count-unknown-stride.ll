; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; ScalarEvolution should be able to compute trip count of the loop by proving
; that this is not an infinite loop with side effects.

; CHECK: Determining loop execution counts for: @foo1
; CHECK: backedge-taken count is ((-1 + %n) /u %s)

; We should have a conservative estimate for the max backedge taken count for
; loops with unknown stride.
; CHECK: max backedge-taken count is -1

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

; Function Attrs: norecurse nounwind
define void @foo1(i32* nocapture %A, i32 %n, i32 %s) #0 {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.05
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %add = add nsw i32 %i.05, %s
  %cmp = icmp slt i32 %add, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}


; Check that we are able to compute trip count of a loop without an entry guard.
; CHECK: Determining loop execution counts for: @foo2
; CHECK: backedge-taken count is ((-1 + (%n smax %s)) /u %s)

; We should have a conservative estimate for the max backedge taken count for
; loops with unknown stride.
; CHECK: max backedge-taken count is -1

; Function Attrs: norecurse nounwind
define void @foo2(i32* nocapture %A, i32 %n, i32 %s) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.05
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %add = add nsw i32 %i.05, %s
  %cmp = icmp slt i32 %add, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

