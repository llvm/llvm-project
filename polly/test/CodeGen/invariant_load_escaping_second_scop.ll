; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true  -polly-process-unprofitable -S < %s | FileCheck %s
;
;    void fence(void);
;
;    void f(int *A, int *B) {
;      int i = 0;
;      int x = 0;
;
;      do {
;        x = *B;
; S:     A[i] += x;
;      } while (i++ < 100);
;
;      fence();
;
;      do {
; P:     A[i]++;
;      } while (i++ < x / 2);
;    }
;
; CHECK: polly.stmt.stmt.P:
; CHECK:   sext i32 %tmp.merge to i64
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, ptr %B) {
entry:
  br label %stmt.S

stmt.S:                                          ; preds = %do.cond, %entry
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %do.cond ], [ 0, %entry ]
  %tmp = load i32, ptr %B, align 4
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv2
  %tmp4 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp4, %tmp
  store i32 %add, ptr %arrayidx, align 4
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv2, 1
  %exitcond = icmp ne i64 %indvars.iv.next3, 101
  br i1 %exitcond, label %stmt.S, label %do.end

do.end:                                           ; preds = %do.cond
  %tmp5 = trunc i64 101 to i32
  call void @fence() #2
  %tmp6 = sext i32 %tmp5 to i64
  br label %stmt.P

stmt.P:                                        ; preds = %do.cond.5, %do.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.cond.5 ], [ %tmp6, %do.end ]
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp7 = load i32, ptr %arrayidx3, align 4
  %inc4 = add nsw i32 %tmp7, 1
  store i32 %inc4, ptr %arrayidx3, align 4
  br label %do.cond.5

do.cond.5:                                        ; preds = %do.body.1
  %div = sdiv i32 %tmp, 2
  %tmp8 = sext i32 %div to i64
  %cmp7 = icmp slt i64 %indvars.iv, %tmp8
  %indvars.iv.next = add i64 %indvars.iv, 1
  br i1 %cmp7, label %stmt.P, label %do.end.8

do.end.8:                                         ; preds = %do.cond.5
  ret void
}

declare void @fence()
