; RUN: opt -loop-unroll -S -mtriple=amdgcn-- -mcpu=tahiti %s | FileCheck %s

; This IR comes from this OpenCL C code:
;
; if (b + 4 > a) {
;   for (int i = 0; i < 4; i++, b++) {
;     if (b + 1 <= a)
;       *(dst + c + b) = 0;
;     else
;       break;
;   }
; }
;
; This test is meant to check that this loop isn't unrolled into more than
; four iterations.  The loop unrolling preferences we currently use cause this
; loop to not be unrolled at all, but that may change in the future.

; CHECK-LABEL: @test
; CHECK: store i8 0, i8 addrspace(1)*
; CHECK-NOT: store i8 0, i8 addrspace(1)*
; CHECK: ret void
define amdgpu_kernel void @test(i8 addrspace(1)* nocapture %dst, i32 %a, i32 %b, i32 %c) {
entry:
  %add = add nsw i32 %b, 4
  %cmp = icmp sgt i32 %add, %a
  br i1 %cmp, label %for.cond.preheader, label %if.end7

for.cond.preheader:                               ; preds = %entry
  %cmp313 = icmp slt i32 %b, %a
  br i1 %cmp313, label %if.then4.lr.ph, label %if.end7.loopexit

if.then4.lr.ph:                                   ; preds = %for.cond.preheader
  %0 = sext i32 %c to i64
  br label %if.then4

if.then4:                                         ; preds = %if.then4.lr.ph, %if.then4
  %i.015 = phi i32 [ 0, %if.then4.lr.ph ], [ %inc, %if.then4 ]
  %b.addr.014 = phi i32 [ %b, %if.then4.lr.ph ], [ %add2, %if.then4 ]
  %add2 = add nsw i32 %b.addr.014, 1
  %1 = sext i32 %b.addr.014 to i64
  %add.ptr.sum = add nsw i64 %1, %0
  %add.ptr5 = getelementptr inbounds i8, i8 addrspace(1)* %dst, i64 %add.ptr.sum
  store i8 0, i8 addrspace(1)* %add.ptr5, align 1
  %inc = add nsw i32 %i.015, 1
  %cmp1 = icmp slt i32 %inc, 4
  %cmp3 = icmp slt i32 %add2, %a
  %or.cond = and i1 %cmp3, %cmp1
  br i1 %or.cond, label %if.then4, label %for.cond.if.end7.loopexit_crit_edge

for.cond.if.end7.loopexit_crit_edge:              ; preds = %if.then4
  br label %if.end7.loopexit

if.end7.loopexit:                                 ; preds = %for.cond.if.end7.loopexit_crit_edge, %for.cond.preheader
  br label %if.end7

if.end7:                                          ; preds = %if.end7.loopexit, %entry
  ret void
}
