; RUN: opt -passes='print<access-info>' -disable-output  < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; FIXME: This should be vectorizable

; void vectorizable_Read_Write(int *A) {
;  for (unsigned i = 1022; i >= 0; i--)
;    A[i+1] = A[i] + 1;
; }

; CHECK: function 'vectorizable_Read_Write':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:     Report: unsafe dependent memory operations in loop
; CHECK-NEXT:     Forward loop carried data dependence that prevents store-to-load forwarding.
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:       ForwardButPreventsForwarding:
; CHECK-NEXT:           %0 = load i32, ptr %arrayidx, align 4 ->
; CHECK-NEXT:           store i32 %add, ptr %gep, align 4

define void @vectorizable_Read_Write(ptr nocapture %A) {
entry:
  %invariant.gep = getelementptr i32, ptr %A, i64 1
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 1022, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %gep = getelementptr i32, ptr %invariant.gep, i64 %indvars.iv
  store i32 %add, ptr %gep, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %cmp.not = icmp eq i64 %indvars.iv, 0
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body
}

