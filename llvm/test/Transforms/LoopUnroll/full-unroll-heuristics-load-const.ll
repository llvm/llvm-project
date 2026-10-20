; RUN: opt < %s -S -passes=loop-unroll -unroll-max-iteration-count-to-analyze=1000 -unroll-threshold=10 -unroll-max-percent-threshold-boost=100 | FileCheck %s -check-prefix=TEST1
; RUN: opt < %s -S -passes=loop-unroll -unroll-max-iteration-count-to-analyze=1000 -unroll-threshold=20 -unroll-max-percent-threshold-boost=200 | FileCheck %s -check-prefix=TEST2
; RUN: opt < %s -S -passes=loop-unroll -unroll-max-iteration-count-to-analyze=1000 -unroll-threshold=20 -unroll-max-percent-threshold-boost=100 | FileCheck %s -check-prefix=TEST3

; This test is a copy of full-unroll-heuristics.ll but with the constant
; wrapped in an extra struct. This should not hinder the analysis.

; If the absolute threshold is too low, we should not unroll:
; TEST1: %array_const_idx = getelementptr inbounds { [9 x i32] }, ptr @known_constant, i64 0, i32 0, i64 %iv

; Otherwise, we should:
; TEST2-NOT: %array_const_idx = getelementptr inbounds { [9 x i32] }, ptr @known_constant, i64 0, i32 0, i64 %iv

; If we do not boost threshold, the unroll will not happen:
; TEST3: %array_const_idx = getelementptr inbounds { [9 x i32] }, ptr @known_constant, i64 0, i32 0, i64 %iv

@known_constant = internal unnamed_addr constant { [9 x i32] } { [9 x i32] [i32 0, i32 -1, i32 0, i32 -1, i32 5, i32 -1, i32 0, i32 -1, i32 0] }, align 16

define i32 @foo(ptr noalias nocapture readonly %src) {
entry:
  br label %loop

loop:                                                ; preds = %loop, %entry
  %iv = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %r  = phi i32 [ 0, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %src, i64 %iv
  %src_element = load i32, ptr %arrayidx, align 4
  %array_const_idx = getelementptr inbounds { [9 x i32] }, ptr @known_constant, i64 0, i32 0, i64 %iv
  %const_array_element = load i32, ptr %array_const_idx, align 4
  %mul = mul nsw i32 %src_element, %const_array_element
  %add = add nsw i32 %mul, %r
  %inc = add nuw nsw i64 %iv, 1
  %exitcond86.i = icmp eq i64 %inc, 9
  br i1 %exitcond86.i, label %loop.end, label %loop

loop.end:                                            ; preds = %loop
  %r.lcssa = phi i32 [ %r, %loop ]
  ret i32 %r.lcssa
}
