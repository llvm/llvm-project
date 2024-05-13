; RUN: opt %loadPolly -basic-aa -polly-print-import-jscop -disable-output < %s | FileCheck %s
;
; Check that we allow the new access functions even though they access
; different locations than the original ones (but the alignment is the
; default, thus there is no problem).
;
; CHECK-DAG: New access function '{ Stmt_for_body[i0] -> MemRef_B[0] }' detected in JSCOP file
; CHECK-DAG: New access function '{ Stmt_for_body[i0] -> MemRef_A[i0] }' detected in JSCOP file
;
;    void simple_stride(int *restrict A, int *restrict B) {
;      for (int i = 0; i < 16; i++)
;        A[i * 2] = B[i * 2];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @simple_stride(ptr noalias %A, ptr noalias %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 16
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = shl nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %tmp
  %tmp4 = load i32, ptr %arrayidx, align 4
  %tmp5 = shl nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %tmp5
  store i32 %tmp4, ptr %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
