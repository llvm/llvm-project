; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; ScalarEvolution should be able to compute trip count of the loop by proving
; that this is not an infinite loop with side effects.

; CHECK-LABEL: Determining loop execution counts for: @foo1
; CHECK: backedge-taken count is ((-1 + (-1 * %s) + (1 umax %s) + %n) /u (1 umax %s))

; We should have a conservative estimate for the max backedge taken count for
; loops with unknown stride.
; CHECK: max backedge-taken count is -1

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"

define void @foo1(i32* nocapture %A, i32 %n, i32 %s) mustprogress {
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
; CHECK-LABEL: Determining loop execution counts for: @foo2
; CHECK: backedge-taken count is ((-1 + (-1 * %s) + (1 umax %s) + (%n smax %s)) /u (1 umax %s))

; We should have a conservative estimate for the max backedge taken count for
; loops with unknown stride.
; CHECK: max backedge-taken count is -1

define void @foo2(i32* nocapture %A, i32 %n, i32 %s) mustprogress {
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

; Check that without mustprogress we don't make assumptions about infinite
; loops being UB.
; CHECK-LABEL: Determining loop execution counts for: @foo3
; CHECK: Loop %for.body: Unpredictable backedge-taken count.
; CHECK: Loop %for.body: Unpredictable max backedge-taken count.

define void @foo3(i32* nocapture %A, i32 %n, i32 %s) {
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

; Same as foo2, but with mustprogress on loop, not function
; CHECK-LABEL: Determining loop execution counts for: @foo4
; CHECK: backedge-taken count is ((-1 + (-1 * %s) + (1 umax %s) + (%n smax %s)) /u (1 umax %s))
; CHECK: max backedge-taken count is -1

define void @foo4(i32* nocapture %A, i32 %n, i32 %s) {
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
  br i1 %cmp, label %for.body, label %for.end, !llvm.loop !8

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; A more complex case with pre-increment compare instead of post-increment.
; CHECK-LABEL: Determining loop execution counts for: @foo5
; CHECK: Loop %for.body: backedge-taken count is ((((-1 * (1 umin ((-1 * %start) + (%n smax %start))))<nuw><nsw> + (-1 * %start) + (%n smax %start)) /u (1 umax %s)) + (1 umin ((-1 * %start) + (%n smax %start))))

; We should have a conservative estimate for the max backedge taken count for
; loops with unknown stride.
; CHECK: max backedge-taken count is -1

define void @foo5(i32* nocapture %A, i32 %n, i32 %s, i32 %start) mustprogress {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %add, %for.body ], [ %start, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.05
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %add = add nsw i32 %i.05, %s
  %cmp = icmp slt i32 %i.05, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
