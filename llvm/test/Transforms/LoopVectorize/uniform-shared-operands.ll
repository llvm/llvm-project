; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=2 -verify-each -S %s | FileCheck %s
; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=4 -verify-each -S %s | FileCheck %s
; REQUIRES: x86-registered-target
;
; Shared operands must not cause exponential uniformity analysis during unrolling.
;
; CHECK-LABEL: define i64 @f(
; CHECK: vector.ph:
; CHECK-COUNT-20: call <2 x i64> @llvm.fshl.v2i64
; CHECK: vector.body:
; CHECK: and <2 x i64>
; CHECK: and <2 x i64>
; CHECK: call i64 @llvm.vector.reduce.xor.v2i64

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @f(i64 %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  %a1 = call i64 @llvm.fshl.i64(i64 %x, i64 %x, i64 %x)
  %a2 = call i64 @llvm.fshl.i64(i64 %a1, i64 %a1, i64 %a1)
  %a3 = call i64 @llvm.fshl.i64(i64 %a2, i64 %a2, i64 %a2)
  %a4 = call i64 @llvm.fshl.i64(i64 %a3, i64 %a3, i64 %a3)
  %a5 = call i64 @llvm.fshl.i64(i64 %a4, i64 %a4, i64 %a4)
  %a6 = call i64 @llvm.fshl.i64(i64 %a5, i64 %a5, i64 %a5)
  %a7 = call i64 @llvm.fshl.i64(i64 %a6, i64 %a6, i64 %a6)
  %a8 = call i64 @llvm.fshl.i64(i64 %a7, i64 %a7, i64 %a7)
  %a9 = call i64 @llvm.fshl.i64(i64 %a8, i64 %a8, i64 %a8)
  %a10 = call i64 @llvm.fshl.i64(i64 %a9, i64 %a9, i64 %a9)
  %a11 = call i64 @llvm.fshl.i64(i64 %a10, i64 %a10, i64 %a10)
  %a12 = call i64 @llvm.fshl.i64(i64 %a11, i64 %a11, i64 %a11)
  %a13 = call i64 @llvm.fshl.i64(i64 %a12, i64 %a12, i64 %a12)
  %a14 = call i64 @llvm.fshl.i64(i64 %a13, i64 %a13, i64 %a13)
  %a15 = call i64 @llvm.fshl.i64(i64 %a14, i64 %a14, i64 %a14)
  %a16 = call i64 @llvm.fshl.i64(i64 %a15, i64 %a15, i64 %a15)
  %a17 = call i64 @llvm.fshl.i64(i64 %a16, i64 %a16, i64 %a16)
  %a18 = call i64 @llvm.fshl.i64(i64 %a17, i64 %a17, i64 %a17)
  %a19 = call i64 @llvm.fshl.i64(i64 %a18, i64 %a18, i64 %a18)
  %a20 = call i64 @llvm.fshl.i64(i64 %a19, i64 %a19, i64 %a19)
  %m = and i64 %a20, %i
  %acc.next = xor i64 %acc, %m
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i64 %acc.next
}

declare i64 @llvm.fshl.i64(i64, i64, i64)
