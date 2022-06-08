; RUN: opt -S -passes='loop(indvars)' -verify-scev < %s | FileCheck %s

; REQUIRES: asserts
; XFAIL: *

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @test_01(i1 %b) {
; CHECK-LABEL: test_01
entry:
  %b.ext = zext i1 %b to i32
  %zero = and i32 %b.ext, 2
  %precond = icmp ne i32 %zero, 0
  call void @llvm.assume(i1 %precond)
  br label %loop

loop:
  %iv = phi i16 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i16 %iv, 1
  %cond = icmp slt i16 %iv, 1
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

define void @test_02() {
; CHECK-LABEL: test_02
entry:
  %tmp = and i1 false, undef
  br i1 %tmp, label %preheader, label %unreached

preheader:                                              ; preds = %entry
  br label %loop

loop:                                              ; preds = %loop, %preheader
  %tmp3 = phi i32 [ 2, %preheader ], [ %tmp4, %loop ]
  %tmp4 = add nuw nsw i32 %tmp3, 1
  %tmp5 = icmp ugt i32 %tmp3, 74
  br i1 %tmp5, label %unreached, label %loop

unreached:                                              ; preds = %loop
  unreachable
}

declare void @llvm.assume(i1)
