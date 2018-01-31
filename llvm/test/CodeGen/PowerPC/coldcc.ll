; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu  < %s | FileCheck %s -check-prefix=COLDCC

define signext i32 @caller(i32 signext %a, i32 signext %b, i32 signext %cold) {
entry:
  %0 = tail call i32 asm "add $0, $1, $2", "=r,r,r,~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"(i32 %a, i32 %b)
  %mul = mul nsw i32 %0, %cold
  %tobool = icmp eq i32 %cold, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %mul1 = mul nsw i32 %mul, %cold
  %mul2 = mul nsw i32 %b, %a
  %call = tail call coldcc signext i32 @callee(i32 signext %a, i32 signext %b)
  %add = add i32 %mul2, %a
  %add3 = add i32 %add, %mul
  %add4 = add i32 %add3, %mul1
  %add5 = add i32 %add4, %call
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %f.0 = phi i32 [ %add5, %if.then ], [ %0, %entry ]
  ret i32 %f.0
}

define internal coldcc signext i32 @callee(i32 signext %a, i32 signext %b) local_unnamed_addr #0 {
entry:
; COLDCC: @callee
; COLDCC: std 6, -8(1)
; COLDCC: std 7, -16(1)
; COLDCC: std 8, -24(1)
; COLDCC: std 9, -32(1)
; COLDCC: std 10, -40(1)
; COLDCC: ld 9, -32(1)
; COLDCC: ld 8, -24(1)
; COLDCC: ld 7, -16(1)
; COLDCC: ld 10, -40(1)
; COLDCC: ld 6, -8(1)
  %0 = tail call i32 asm "add $0, $1, $2", "=r,r,r,~{r6},~{r7},~{r8},~{r9},~{r10}"(i32 %a, i32 %b)
  %mul = mul nsw i32 %a, 3
  %1 = mul i32 %b, -5
  %add = add i32 %1, %mul
  %sub = add i32 %add, %0
  ret i32 %sub
}

attributes #0 = { noinline }
