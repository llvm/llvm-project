; RUN: opt < %s -S -passes=loop-unroll -unroll-max-iteration-count-to-analyze=100 -unroll-threshold=12 -unroll-max-percent-threshold-boost=400 | FileCheck %s
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(loop-unroll-full)' -unroll-max-iteration-count-to-analyze=100 -unroll-threshold=12 -unroll-max-percent-threshold-boost=400 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@known_constant = internal unnamed_addr constant [10 x i32] [i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0], align 16

; If a load becomes a constant after loop unrolling, we sometimes can simplify
; CFG. This test verifies that we handle such cases.
; After one operand in an instruction is constant-folded and the
; instruction is simplified, the other operand might become dead.
; In this test we have::
; for i in 1..10:
;   r += Aptr B[i]
; A[i] is 0 almost at every iteration, so there is no need in loading B[i] at
; all.


; CHECK-LABEL: @unroll_dce
; CHECK-NOT:   br i1 %exitcond, label %for.end, label %for.body
define i32 @unroll_dce(ptr noalias nocapture readonly %b) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv.0 = phi i64 [ 0, %entry ], [ %iv.1, %for.body ]
  %r.0 = phi i32 [ 0, %entry ], [ %r.1, %for.body ]
  %arrayidx1 = getelementptr inbounds [10 x i32], ptr @known_constant, i64 0, i64 %iv.0
  %x1 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %iv.0
  %x2 = load i32, ptr %arrayidx2, align 4
  %mul = mul i32 %x1, %x2
  %r.1 = add i32 %mul, %r.0
  %iv.1 = add nuw nsw i64 %iv.0, 1
  %exitcond = icmp eq i64 %iv.1, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 %r.1
}
