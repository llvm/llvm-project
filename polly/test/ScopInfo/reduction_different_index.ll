; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
; Verify if the following case is not detected as reduction.
;
; void f(int *A,int *sum) {
;   for (int i = 0; i < 1024; i++)
;     sum[0] = sum[1] + A[i];
; }
;
; Verify that we don't detect the reduction on sum
;
; CHECK: ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT: { Stmt_for_body[i0] -> MemRef_sum[1] };
; CHECK-NEXT:ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT: { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK-NEXT:MustWriteAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT: { Stmt_for_body[i0] -> MemRef_sum[0] };
;
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define dso_local void @f(ptr nocapture noundef readonly %A, ptr nocapture noundef %sum) local_unnamed_addr #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry.split, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %sum, i64 1
  %0 = load i32, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx1
  %add = add nsw i32 %1, %0
  store i32 %add, ptr %sum
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
