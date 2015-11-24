; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
;    void jd(int *A, int *B, int c) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = B[c - 10] + B[5];
;    }
;
; CHECK:  sext i32 %c to i64
; CHECK:  sext i32 %c to i64
; CHECK:  %[[M0:[._a-zA-Z0-9]*]] = sext i32 %c to i64
; CHECK:  %[[M1:[._a-zA-Z0-9]*]] = icmp sle i64 %[[M0]], 15
; CHECK:  %[[M2:[._a-zA-Z0-9]*]] = sext i32 %c to i64
; CHECK:  %[[M3:[._a-zA-Z0-9]*]] = sub nsw i64 %[[M2]], 9
; CHECK:  %[[M4:[._a-zA-Z0-9]*]] = select i1 %[[M1]], i64 6, i64 %[[M3]]
; CHECK:  %[[BMax:[._a-zA-Z0-9]*]] = getelementptr i32, i32* %B, i64 %[[M4]]
; CHECK:  %[[AMin:[._a-zA-Z0-9]*]] = getelementptr i32, i32* %A, i64 0
; CHECK:  %[[BltA:[._a-zA-Z0-9]*]] = icmp ule i32* %[[BMax]], %[[AMin]]
; CHECK:  %[[AMax:[._a-zA-Z0-9]*]] = getelementptr i32, i32* %A, i64 1024
; CHECK:  %[[m0:[._a-zA-Z0-9]*]] = sext i32 %c to i64
; CHECK:  %[[m1:[._a-zA-Z0-9]*]] = icmp sge i64 %[[m0]], 15
; CHECK:  %[[m2:[._a-zA-Z0-9]*]] = sext i32 %c to i64
; CHECK:  %[[m3:[._a-zA-Z0-9]*]] = sub nsw i64 %[[m2]], 10
; CHECK:  %[[m4:[._a-zA-Z0-9]*]] = select i1 %[[m1]], i64 5, i64 %[[m3]]
; CHECK:  %[[BMin:[._a-zA-Z0-9]*]] = getelementptr i32, i32* %B, i64 %[[m4]]
; CHECK:  %[[AltB:[._a-zA-Z0-9]*]] = icmp ule i32* %[[AMax]], %[[BMin]]
; CHECK:  %[[NoAlias:[._a-zA-Z0-9]*]] = or i1 %[[BltA]], %[[AltB]]
; CHECK:  %[[RTC:[._a-zA-Z0-9]*]] = and i1 %3, %[[NoAlias]]
; CHECK:  br i1 %[[RTC]], label %polly.start, label %for.cond
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32* %B, i32 %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %sub = add nsw i32 %c, -10
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %idxprom
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %B, i64 5
  %tmp1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %tmp, %tmp1
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
