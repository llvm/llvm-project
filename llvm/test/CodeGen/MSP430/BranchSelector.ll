; RUN: llc < %s -march=msp430 | FileCheck %s
target datalayout = "e-m:e-p:16:16-i32:16:32-a:16-n8:16"
target triple = "msp430"

@reg = common global i16 0, align 2

define void @WriteBurstPATable(i16 %count) #0 {
entry:
  br label %while.cond

while.cond:
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  %v0 = load volatile i16, ptr @reg, align 2
  %lnot = icmp eq i16 %v0, 0

; This BB should be split and all branches should be expanded.
; CHECK-LABEL: .LBB0_1:
; CHECK: jne	.LBB0_2
; CHECK: br	#.LBB0_1
; CHECK: .LBB0_2:
; CHECK: br	#.LBB0_4
; CHECK: .LBB0_3:

  br i1 %lnot, label %while.cond, label %while.end

while.end:
  %i.0.i.0.1822 = load volatile i16, ptr @reg, align 1
  %cmp23 = icmp ult i16 %i.0.i.0.1822, %count
  br i1 %cmp23, label %for.body, label %for.end

for.body:
  br label %while.cond6

while.cond6:
  %0 = load volatile i16, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 19, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  br label %for.inc

for.inc:
  %1 = load volatile i16, ptr @reg, align 2
  %cmp = icmp ult i16 %1, %count

; This branch should be expanded.
; CHECK-LABEL: .LBB0_4:
; CHECK: jhs	.LBB0_5
; CHECK: br	#.LBB0_3
; CHECK: .LBB0_5:

  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @WriteSinglePATable() #0 {
entry:
  br label %begin
begin:
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  store volatile i16 13, ptr @reg, align 2
  store volatile i16 17, ptr @reg, align 2
  store volatile i16 11, ptr @reg, align 2
  %v2 = load volatile i16, ptr @reg, align 2
  %lnot = icmp eq i16 %v2, 0

; This branch should not be expanded
; CHECK-LABEL: .LBB1_1:
; CHECK: jeq	.LBB1_1
; CHECK: %bb.2:
; CHECK: ret
  br i1 %lnot, label %begin, label %end

end:
  ret void
}
