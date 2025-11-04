; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
;    void jd(int *A, int *B, int c) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = B[c - 10] + B[5];
;    }
;
; CHECK:  %[[Ctx:[._a-zA-Z0-9]*]] = and i1 true
; CHECK-NEXT:  %[[M0:[._a-zA-Z0-9]*]] = sext i32 %c to i64
; CHECK-NEXT:  %[[M3:[._a-zA-Z0-9]*]] = call { i64, i1 } @llvm.ssub.with.overflow.i64(i64 %[[M0]], i64 9)
; CHECK-NEXT:  %[[M3O:[._a-zA-Z0-9]*]] = extractvalue { i64, i1 } %[[M3]], 1
; CHECK-NEXT:  %[[OS0:[._a-zA-Z0-9]*]]   = or i1 false, %[[M3O]]
; CHECK-NEXT:  %[[M3R:[._a-zA-Z0-9]*]] = extractvalue { i64, i1 } %[[M3]], 0
; CHECK-NEXT:  %[[M1:[._a-zA-Z0-9]*]] = icmp sgt i64 6, %[[M3R]]
; CHECK-NEXT:  %[[M4:[._a-zA-Z0-9]*]] = select i1 %[[M1]], i64 6, i64 %[[M3R]]
; CHECK-NEXT:  %[[BMax:[._a-zA-Z0-9]*]] = getelementptr i32, ptr %B, i64 %[[M4]]
; CHECK-NEXT:  %[[AMin:[._a-zA-Z0-9]*]] = getelementptr i32, ptr %A, i64 0
; CHECK-NEXT:  %[[BMaxI:[._a-zA-Z0-9]*]] = ptrtoint ptr %[[BMax]] to i64
; CHECK-NEXT:  %[[AMinI:[._a-zA-Z0-9]*]] = ptrtoint ptr %[[AMin]] to i64
; CHECK-NEXT:  %[[BltA:[._a-zA-Z0-9]*]] = icmp ule i64 %[[BMaxI]], %[[AMinI]]
; CHECK-NEXT:  %[[AMax:[._a-zA-Z0-9]*]] = getelementptr i32, ptr %A, i64 1024
; CHECK-NEXT:  %[[m0:[._a-zA-Z0-9]*]] = sext i32 %c to i64
; CHECK-NEXT:  %[[m3:[._a-zA-Z0-9]*]] = call { i64, i1 } @llvm.ssub.with.overflow.i64(i64 %[[m0]], i64 10)
; CHECK-NEXT:  %[[m3O:[._a-zA-Z0-9]*]] = extractvalue { i64, i1 } %[[m3]], 1
; CHECK-NEXT:  %[[OS1:[._a-zA-Z0-9]*]]   = or i1 %[[OS0]], %[[m3O]]
; CHECK-NEXT:  %[[m3R:[._a-zA-Z0-9]*]] = extractvalue { i64, i1 } %[[m3]], 0
; CHECK-NEXT:  %[[m1:[._a-zA-Z0-9]*]] = icmp slt i64 5, %[[m3R]]
; CHECK-NEXT:  %[[m4:[._a-zA-Z0-9]*]] = select i1 %[[m1]], i64 5, i64 %[[m3R]]
; CHECK-NEXT:  %[[BMin:[._a-zA-Z0-9]*]] = getelementptr i32, ptr %B, i64 %[[m4]]
; CHECK-NEXT:  %[[AMaxI:[._a-zA-Z0-9]*]] = ptrtoint ptr %[[AMax]] to i64
; CHECK-NEXT:  %[[BMinI:[._a-zA-Z0-9]*]] = ptrtoint ptr %[[BMin]] to i64
; CHECK-NEXT:  %[[AltB:[._a-zA-Z0-9]*]] = icmp ule i64 %[[AMaxI]], %[[BMinI]]
; CHECK-NEXT:  %[[NoAlias:[._a-zA-Z0-9]*]] = or i1 %[[BltA]], %[[AltB]]
; CHECK-NEXT:  %[[RTC:[._a-zA-Z0-9]*]] = and i1 %[[Ctx]], %[[NoAlias]]
; CHECK-NEXT:  %[[NOV:[._a-zA-Z0-9]*]] = xor i1 %[[OS1]], true
; CHECK-NEXT:  %[[RTCV:[._a-zA-Z0-9]*]] = and i1 %[[RTC]], %[[NOV]]
; CHECK-NEXT:  br i1 %[[RTCV]], label %polly.start, label %for.cond
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
  %sub = add nsw i32 %c, -10
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %idxprom
  %tmp = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %B, i64 5
  %tmp1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %tmp, %tmp1
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
