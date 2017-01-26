; RUN: opt < %s -S -loop-unroll -unroll-max-iteration-count-to-analyze=1000 -unroll-threshold=10 -unroll-max-percent-threshold-boost=200 | FileCheck %s
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(unroll-full)' -unroll-max-iteration-count-to-analyze=1000 -unroll-threshold=10 -unroll-max-percent-threshold-boost=200 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@unknown_global = internal unnamed_addr global [9 x i32] [i32 0, i32 -1, i32 0, i32 -1, i32 5, i32 -1, i32 0, i32 -1, i32 0], align 16
@weak_constant = weak unnamed_addr constant [9 x i32] [i32 0, i32 -1, i32 0, i32 -1, i32 5, i32 -1, i32 0, i32 -1, i32 0], align 16

; Though @unknown_global is initialized with constant values, we can't consider
; it as a constant, so we shouldn't unroll the loop.
; CHECK-LABEL: @foo
; CHECK: %array_const_idx = getelementptr inbounds [9 x i32], [9 x i32]* @unknown_global, i64 0, i64 %iv
define i32 @foo(i32* noalias nocapture readonly %src) {
entry:
  br label %loop

loop:                                                ; preds = %loop, %entry
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %r  = phi i32 [ 0, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %src, i64 %iv
  %src_element = load i32, i32* %arrayidx, align 4
  %array_const_idx = getelementptr inbounds [9 x i32], [9 x i32]* @unknown_global, i64 0, i64 %iv
  %const_array_element = load i32, i32* %array_const_idx, align 4
  %mul = mul nsw i32 %src_element, %const_array_element
  %add = add nsw i32 %mul, %r
  %inc = add nuw nsw i64 %iv, 1
  %exitcond86.i = icmp eq i64 %inc, 9
  br i1 %exitcond86.i, label %loop.end, label %loop

loop.end:                                            ; preds = %loop
  %r.lcssa = phi i32 [ %r, %loop ]
  ret i32 %r.lcssa
}

; Similarly, we can't consider 'weak' symbols as a known constant value, so we
; shouldn't unroll the loop.
; CHECK-LABEL: @foo2
; CHECK: %array_const_idx = getelementptr inbounds [9 x i32], [9 x i32]* @weak_constant, i64 0, i64 %iv
define i32 @foo2(i32* noalias nocapture readonly %src) {
entry:
  br label %loop

loop:                                                ; preds = %loop, %entry
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %r  = phi i32 [ 0, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %src, i64 %iv
  %src_element = load i32, i32* %arrayidx, align 4
  %array_const_idx = getelementptr inbounds [9 x i32], [9 x i32]* @weak_constant, i64 0, i64 %iv
  %const_array_element = load i32, i32* %array_const_idx, align 4
  %mul = mul nsw i32 %src_element, %const_array_element
  %add = add nsw i32 %mul, %r
  %inc = add nuw nsw i64 %iv, 1
  %exitcond86.i = icmp eq i64 %inc, 9
  br i1 %exitcond86.i, label %loop.end, label %loop

loop.end:                                            ; preds = %loop
  %r.lcssa = phi i32 [ %r, %loop ]
  ret i32 %r.lcssa
}

; In this case the loaded value is used only to control branch.
; If we missed that, we could've thought that it's unused and unrolling would
; clean up almost entire loop. Make sure that we do not unroll such loop.
; CHECK-LABEL: @foo3
; CHECK: br i1 %exitcond, label %loop.end, label %loop.header
define i32 @foo3(i32* noalias nocapture readonly %src) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop.latch ]
  %r1  = phi i32 [ 0, %entry ], [ %r3, %loop.latch ]
  %arrayidx = getelementptr inbounds i32, i32* %src, i64 %iv
  %src_element = load i32, i32* %arrayidx, align 4
  %cmp = icmp eq i32 0, %src_element
  br i1 %cmp, label %loop.if, label %loop.latch

loop.if:
  %r2 = add i32 %r1, 1
  br label %loop.latch

loop.latch:
  %r3 = phi i32 [%r1, %loop.header], [%r2, %loop.if]
  %inc = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %inc, 9
  br i1 %exitcond, label %loop.end, label %loop.header

loop.end:
  %r.lcssa = phi i32 [ %r3, %loop.latch ]
  ret i32 %r.lcssa
}
