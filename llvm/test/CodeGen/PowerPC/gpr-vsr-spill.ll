; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu -ppc-enable-gpr-to-vsr-spills  < %s | FileCheck %s
define signext i32 @foo(i32 signext %a, i32 signext %b) {
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = tail call i32 asm "add $0, $1, $2", "=r,r,r,~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29}"(i32 %a, i32 %b)
  %mul = mul nsw i32 %0, %a
  %add = add i32 %b, %a
  %tmp = add i32 %add, %mul
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %e.0 = phi i32 [ %tmp, %if.then ], [ undef, %entry ]
  ret i32 %e.0
; CHECK: @foo
; CHECK: mr [[NEWREG:[0-9]+]], 3
; CHECK: mr [[REG1:[0-9]+]], 4
; CHECK: mtvsrd [[NEWREG2:[0-9]+]], 4
; CHECK: add {{[0-9]+}}, [[NEWREG]], [[REG1]]
; CHECK: mffprd [[REG2:[0-9]+]], [[NEWREG2]]
; CHECK: add {{[0-9]+}}, [[REG2]], [[NEWREG]]
}
