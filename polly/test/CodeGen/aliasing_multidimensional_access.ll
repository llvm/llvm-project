; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
; Check that we calculate the maximal access into array A correctly and track the overflow state.
;
; CHECK:  %[[TMP0:[._0-9a-zA-Z]*]]  = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 99, i64 %m)
; CHECK:  %[[TMP0O:[._0-9a-zA-Z]*]] = extractvalue { i64, i1 } %[[TMP0]], 1
; CHECK:  %[[OS0:[._0-9a-zA-Z]*]]   = or i1 {{[^,]*}}, %[[TMP0O]]
; CHECK:  %[[TMP0R:[._0-9a-zA-Z]*]] = extractvalue { i64, i1 } %[[TMP0]], 0
; CHECK:  %[[TMP1:[._0-9a-zA-Z]*]]  = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %[[TMP0R]], i64 149)
; CHECK:  %[[TMP1O:[._0-9a-zA-Z]*]] = extractvalue { i64, i1 } %[[TMP1]], 1
; CHECK:  %[[OS1:[._0-9a-zA-Z]*]]   = or i1 %[[OS0]], %[[TMP1O]]
; CHECK:  %[[TMP1R:[._0-9a-zA-Z]*]] = extractvalue { i64, i1 } %[[TMP1]], 0
; CHECK:  %[[TMP2:[._0-9a-zA-Z]*]]  = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 %[[TMP1R]], i64 %p)
; CHECK:  %[[TMP2O:[._0-9a-zA-Z]*]] = extractvalue { i64, i1 } %[[TMP2]], 1
; CHECK:  %[[OS2:[._0-9a-zA-Z]*]]   = or i1 %[[OS1]], %[[TMP2O]]
; CHECK:  %[[TMP2R:[._0-9a-zA-Z]*]] = extractvalue { i64, i1 } %[[TMP2]], 0
; CHECK:  %[[TMP3:[._0-9a-zA-Z]*]]  = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %[[TMP2R]], i64 150)
; CHECK:  %[[TMP3O:[._0-9a-zA-Z]*]] = extractvalue { i64, i1 } %[[TMP3]], 1
; CHECK:  %[[OS3:[._0-9a-zA-Z]*]]   = or i1 %[[OS2]], %[[TMP3O]]
; CHECK:  %[[TMP3R:[._0-9a-zA-Z]*]] = extractvalue { i64, i1 } %[[TMP3]], 0
; CHECK:  %polly.access.A{{[0-9]*}} = getelementptr double, ptr %A, i64 %[[TMP3R]]
;
; CHECK:  %polly.rtc.overflown = xor i1 %[[OS3]], true
; CHECK:  %polly.rtc.result = and i1 %{{[^,]*}}, %polly.rtc.overflown
; CHECK:  br i1 %polly.rtc.result,
;
;    void foo(long n, long m, long p, double A[n][m][p], int *B) {
;      for (long i = 0; i < 100; i++)
;        for (long j = 0; j < 150; j++)
;          for (long k = 0; k < 150; k++)
;            A[i][j][k] = B[k];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %n, i64 %m, i64 %p, ptr %A, ptr %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc03, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc04, %for.inc03 ]
  %exitcond2 = icmp ne i64 %i.0, 100
  br i1 %exitcond2, label %for.body, label %for.end15

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc00, %for.body
  %j.0 = phi i64 [ 0, %for.body ], [ %inc01, %for.inc00 ]
  %exitcond1 = icmp ne i64 %j.0, 150
  br i1 %exitcond1, label %for.body3, label %for.end12

for.body3:                                        ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i64 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %k.0, 150
  br i1 %exitcond, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %k.0
  %tmp3 = load i32, ptr %arrayidx, align 2
  %conv = sitofp i32 %tmp3 to double
  %tmp4 = mul nuw i64 %m, %p
  %tmp5 = mul nsw i64 %i.0, %tmp4
  %tmp6 = mul nsw i64 %j.0, %p
  %arrayidx7.sum = add i64 %tmp5, %tmp6
  %arrayidx8.sum = add i64 %arrayidx7.sum, %k.0
  %arrayidx9 = getelementptr inbounds double, ptr %A, i64 %arrayidx8.sum
  store double %conv, ptr %arrayidx9, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %inc = add nsw i64 %k.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc00

for.inc00:                                        ; preds = %for.end
  %inc01 = add nsw i64 %j.0, 1
  br label %for.cond1

for.end12:                                        ; preds = %for.cond1
  br label %for.inc03

for.inc03:                                        ; preds = %for.end12
  %inc04 = add nsw i64 %i.0, 1
  br label %for.cond

for.end15:                                        ; preds = %for.cond
  ret void
}
