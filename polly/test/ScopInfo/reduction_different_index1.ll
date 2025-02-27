; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
; Verify if the following case is not detected as reduction.
;
; void f(int *A, int *sum, int i1, int i2) {
;   for (int i = 0; i < 1024; i++)
;     sum[i2] = sum[i1] + A[i];
; }
;
; Verify that we don't detect the reduction on sum
;
; CHECK: ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT: { Stmt_for_body[i0] -> MemRef_sum[i1] };
; CHECK-NEXT:ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT: { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK-NEXT:MustWriteAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT: { Stmt_for_body[i0] -> MemRef_sum[i2] };
;
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @f(ptr nocapture noundef readonly %A, ptr nocapture noundef %sum, i32 noundef %i1, i32 noundef %i2) local_unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %idxprom = sext i32 %i1 to i64
  %arrayidx = getelementptr inbounds i32, ptr %sum, i64 %idxprom
  %idxprom3 = sext i32 %i2 to i64
  %arrayidx4 = getelementptr inbounds i32, ptr %sum, i64 %idxprom3
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry.split, %for.body
  %indvars.iv = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.body ]
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
