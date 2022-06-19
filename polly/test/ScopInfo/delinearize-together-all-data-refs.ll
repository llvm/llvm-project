; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s

; void foo(long n, long m, long o, double A[n][m][o]) {
;   for (long i = 0; i < n-3; i++)
;     for (long j = 4; j < m; j++)
;       for (long k = 0; k < o-7; k++) {
;         A[i+3][j-4][k+7] = 1.0;
;         A[i][0][k] = 2.0;
;       }
; }


; CHECK: Arrays {
; CHECK:     double MemRef_A[*][%m][%o]; // Element size 8
; CHECK: }

; CHECK: [m, o, n] -> { Stmt_for_body6[i0, i1, i2] -> MemRef_A[3 + i0, i1, 7 + i2] };
; CHECK: [m, o, n] -> { Stmt_for_body6[i0, i1, i2] -> MemRef_A[i0, 0, i2] };

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @foo(i64 %n, i64 %m, i64 %o, double* nocapture %A) {
entry:
  %cmp35 = icmp sgt i64 %n, 0
  br i1 %cmp35, label %for.cond1.preheader.lr.ph, label %for.end18

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp233 = icmp sgt i64 %m, 0
  %cmp531 = icmp sgt i64 %o, 0
  %0 = mul nuw i64 %o, %m
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc16, %for.cond1.preheader.lr.ph
  %i.036 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %inc17, %for.inc16 ]
  br i1 %cmp233, label %for.cond4.preheader.lr.ph, label %for.inc16

for.cond4.preheader.lr.ph:                        ; preds = %for.cond1.preheader
  %add7 = add nsw i64 %i.036, 3
  %1 = mul nsw i64 %add7, %0
  %add = add i64 %1, 7
  %2 = mul nsw i64 %i.036, %0
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc13, %for.cond4.preheader.lr.ph
  %j.034 = phi i64 [ 4, %for.cond4.preheader.lr.ph ], [ %inc14, %for.inc13 ]
  br i1 %cmp531, label %for.body6.lr.ph, label %for.inc13

for.body6.lr.ph:                                  ; preds = %for.cond4.preheader
  %sub = add nsw i64 %j.034, -4
  %3 = mul nsw i64 %sub, %o
  %arrayidx.sum = add i64 %add, %3
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body6.lr.ph
  %k.032 = phi i64 [ 0, %for.body6.lr.ph ], [ %inc, %for.body6 ]
  %arrayidx8.sum = add i64 %arrayidx.sum, %k.032
  %arrayidx9 = getelementptr inbounds double, double* %A, i64 %arrayidx8.sum
  store double 1.000000e+00, double* %arrayidx9, align 8
  %arrayidx10.sum = add i64 %k.032, %2
  %arrayidx12 = getelementptr inbounds double, double* %A, i64 %arrayidx10.sum
  store double 2.000000e+00, double* %arrayidx12, align 8
  %inc = add nsw i64 %k.032, 1
  %osub = sub nsw i64 %o, 7
  %exitcond = icmp eq i64 %inc, %osub
  br i1 %exitcond, label %for.inc13, label %for.body6

for.inc13:                                        ; preds = %for.body6, %for.cond4.preheader
  %inc14 = add nsw i64 %j.034, 1
  %exitcond37 = icmp eq i64 %inc14, %m
  br i1 %exitcond37, label %for.inc16, label %for.cond4.preheader

for.inc16:                                        ; preds = %for.inc13, %for.cond1.preheader
  %inc17 = add nsw i64 %i.036, 1
  %nsub = sub nsw i64 %n, 3
  %exitcond38 = icmp eq i64 %inc17, %nsub
  br i1 %exitcond38, label %for.end18, label %for.cond1.preheader

for.end18:                                        ; preds = %for.inc16, %entry
  ret void
}
