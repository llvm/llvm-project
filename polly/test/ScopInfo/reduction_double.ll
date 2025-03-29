; RUN: opt %loadPolly -basic-aa -polly-print-scops -disable-output -polly-allow-nonaffine < %s | FileCheck %s
;
; Verify if two independent reductions in same loop is detected
;
; CHECK: Stmt_for_body
; CHECK: Reduction Type: +
; CHECK-NEXT: MemRef_sum1[0]
; CHECK-NEXT: Reduction Type: +
; CHECK-NEXT: MemRef_sum1[0]
;
; CHECK: Stmt_for_body_b
; CHECK: Reduction Type: +
; CHECK-NEXT: MemRef_sum2[0]
; CHECK-NEXT: Reduction Type: +
; CHECK-NEXT: MemRef_sum2[0]
;
; int red(int *A, int *B, int *sum, int * prod, int n) {
;   for (int i = 0; i < n; ++i) {
;     *sum += A[i];
;     *prod += B[i];
;   }
; }

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local i32 @red(ptr nocapture noundef readonly %A, ptr nocapture noundef readonly %B, ptr nocapture noundef %sum1, ptr nocapture noundef %sum2, i32 noundef %n) local_unnamed_addr #0 {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret i32 undef

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx
  %1 = load i32, ptr %sum1
  %add = add nsw i32 %1, %0
  store i32 %add, ptr %sum1
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx2
  %3 = load i32, ptr %sum2
  %add3 = add nsw i32 %3, %2
  store i32 %add3, ptr %sum2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}


