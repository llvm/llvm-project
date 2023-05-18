; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
;    void jd(int *A, int *B, int c) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = B[c];
;    }
;
; CHECK:  %[[Cext:[._a-zA-Z0-9]*]] = sext i32 %c to i64
; CHECK:  %[[Cp1:[._a-zA-Z0-9]*]]  = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %[[Cext]], i64 1)
; CHECK:  %[[Cp1O:[._a-zA-Z0-9]*]] = extractvalue { i64, i1 } %[[Cp1]], 1
; CHECK:  %[[OS:[._a-zA-Z0-9]*]]   = or i1 false, %[[Cp1O]]
; CHECK:  %[[Cp1R:[._a-zA-Z0-9]*]] = extractvalue { i64, i1 } %[[Cp1]], 0
; CHECK:  %[[BMax:[._a-zA-Z0-9]*]] = getelementptr i32, ptr %B, i64 %[[Cp1R]]
; CHECK:  %[[AMin:[._a-zA-Z0-9]*]] = getelementptr i32, ptr %A, i64 0
; CHECK:  %[[BMaxI:[._a-zA-Z0-9]*]] = ptrtoint ptr %[[BMax]] to i64
; CHECK:  %[[AMinI:[._a-zA-Z0-9]*]] = ptrtoint ptr %[[AMin]] to i64
; CHECK:  %[[BltA:[._a-zA-Z0-9]*]] = icmp ule i64 %[[BMaxI]], %[[AMinI]]
; CHECK:  %[[AMax:[._a-zA-Z0-9]*]] = getelementptr i32, ptr %A, i64 1024
; CHECK:  %[[BMin:[._a-zA-Z0-9]*]] = getelementptr i32, ptr %B, i32 %c
; CHECK:  %[[AMaxI:[._a-zA-Z0-9]*]] = ptrtoint ptr %[[AMax]] to i64
; CHECK:  %[[BMinI:[._a-zA-Z0-9]*]] = ptrtoint ptr %[[BMin]] to i64
; CHECK:  %[[AltB:[._a-zA-Z0-9]*]] = icmp ule i64 %[[AMaxI]], %[[BMinI]]
; CHECK:  %[[NoAlias:[._a-zA-Z0-9]*]] = or i1 %[[BltA]], %[[AltB]]
; CHECK:  %[[RTC:[._a-zA-Z0-9]*]] = and i1 true, %[[NoAlias]]
; CHECK:  %[[NOV:[._a-zA-Z0-9]*]] = xor i1 %[[OS]], true
; CHECK:  %[[RTCV:[._a-zA-Z0-9]*]] = and i1 %[[RTC]], %[[NOV]]
; CHECK:  br i1 %[[RTCV]], label %polly.start, label %for.cond
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(ptr %A, ptr %B, i32 %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %c to i64
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
  %tmp = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %tmp, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
