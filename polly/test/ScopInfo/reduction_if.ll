; RUN: opt %loadPolly -basic-aa -polly-print-scops -disable-output -polly-allow-nonaffine < %s | FileCheck %s
;
; Verify if reduction spread across multiple blocks in a single scop statement are detected
;
; CHECK: Stmt_for_body
; CHECK: Reduction Type: +
; CHECK-NEXT: MemRef_sum[0]
; CHECK: Reduction Type: +
; CHECK-NEXT: MemRef_sum[0]
;
; void f(int*__restrict A, int*__restrict B, int *sum) {
;   for (int i = 0; i < 4444; ++i) {
;     if (B[i])
;       *sum += A[i];
;   }
; }

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @f(ptr noalias nocapture noundef readonly %A, ptr noalias nocapture noundef readonly %B, ptr nocapture noundef %sum) local_unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  ret void

for.body:                                         ; preds = %entry.split, %for.inc
  %indvars.iv = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2
  %2 = load i32, ptr %sum
  %add = add nsw i32 %2, %1
  store i32 %add, ptr %sum
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4444
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

