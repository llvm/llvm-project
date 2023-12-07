; RUN: opt %loadPolly -polly-import-jscop -polly-print-ast -polly-ast-detect-parallel -disable-output < %s | FileCheck %s
;
; CHECK:    #pragma known-parallel
; CHECK:    for (int c0 = 0; c0 <= 1; c0 += 1)
; CHECK:      for (int c1 = c0; c1 < 2 * n; c1 += 2)
; CHECK:        #pragma simd reduction
; CHECK:        for (int c3 = 0; c3 <= 1023; c3 += 1)
; CHECK:          Stmt_for_body3(c1, c3);
;
;    void rmsmd(int *A, long n) {
;      for (long i = 0; i < 2 * n; i++)
;        for (long j = 0; j < 1024; j++)
;          A[i] += i;
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @rmsmd(ptr %A, i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc4, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc5, %for.inc4 ]
  %mul = shl nsw i32 %n, 1
  %cmp = icmp slt i32 %i.0, %mul
  br i1 %cmp, label %for.body, label %for.end6

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.0
  %tmp = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp, %i.0
  store i32 %add, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc4

for.inc4:                                         ; preds = %for.end
  %inc5 = add nsw i32 %i.0, 1
  br label %for.cond

for.end6:                                         ; preds = %for.cond
  ret void
}
