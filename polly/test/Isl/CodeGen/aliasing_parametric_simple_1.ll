; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
;    void jd(int *A, int *B, int c) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = B[c];
;    }
;
; CHECK:  %[[Cext:[._a-zA-Z0-9]*]] = sext i32 %c to i33
; CHECK:  %[[Cp1:[._a-zA-Z0-9]*]]  = add nsw i33 %[[Cext]], 1
; CHECK:  %[[BMax:[._a-zA-Z0-9]*]] = getelementptr i32, i32* %B, i33 %[[Cp1]]
; CHECK:  %[[AMin:[._a-zA-Z0-9]*]] = getelementptr i32, i32* %A, i1 false
; CHECK:  %[[BMaxI:[._a-zA-Z0-9]*]] = ptrtoint i32* %[[BMax]] to i64
; CHECK:  %[[AMinI:[._a-zA-Z0-9]*]] = ptrtoint i32* %[[AMin]] to i64
; CHECK:  %[[BltA:[._a-zA-Z0-9]*]] = icmp ule i64 %[[BMaxI]], %[[AMinI]]
; CHECK:  %[[AMax:[._a-zA-Z0-9]*]] = getelementptr i32, i32* %A, i12 1024
; CHECK:  %[[BMin:[._a-zA-Z0-9]*]] = getelementptr i32, i32* %B, i32 %c
; CHECK:  %[[AMaxI:[._a-zA-Z0-9]*]] = ptrtoint i32* %[[AMax]] to i64
; CHECK:  %[[BMinI:[._a-zA-Z0-9]*]] = ptrtoint i32* %[[BMin]] to i64
; CHECK:  %[[AltB:[._a-zA-Z0-9]*]] = icmp ule i64 %[[AMaxI]], %[[BMinI]]
; CHECK:  %[[NoAlias:[._a-zA-Z0-9]*]] = or i1 %[[BltA]], %[[AltB]]
; CHECK:  %[[RTC:[._a-zA-Z0-9]*]] = and i1 true, %[[NoAlias]]
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
  %idxprom = sext i32 %c to i64
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %idxprom
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %tmp, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
