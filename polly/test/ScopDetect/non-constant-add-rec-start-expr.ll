; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s

; CHECK: Valid Region for Scop: bb11 => bb25

; Ensure that this test case does not trigger an assertion. At some point,
; we asserted on scops containing accesses where the access function contained
; an AddRec expression with a non-constant step expression. This got missed, as
; this very specific pattern does not seem too common. Even in this test case,
; it disappears as soon as we turn the infinite loop into a finite loop.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hoge() local_unnamed_addr {
bb:
  %tmp = alloca [18 x [16 x i32]]
  %tmp1 = alloca [17 x i32]
  br label %bb2

bb2:
  %tmp3 = phi i64 [ 0, %bb ], [ %tmp5, %bb2 ]
  %tmp4 = add nuw nsw i64 %tmp3, 2
  %tmp5 = add nuw nsw i64 %tmp3, 1
  br i1 undef, label %bb2, label %bb11

bb11:
  %tmp12 = phi i64 [ %tmp23, %bb24 ], [ 1, %bb2 ]
  %tmp14 = getelementptr inbounds [17 x i32], ptr %tmp1, i64 0, i64 1
  br label %bb15

bb15:
  %tmp16 = sub nsw i64 %tmp12, 1
  %tmp17 = shl i64 %tmp16, 32
  %tmp18 = ashr exact i64 %tmp17, 32
  %tmp19 = getelementptr inbounds [18 x [16 x i32]], ptr %tmp, i64 0, i64 %tmp4, i64 %tmp18
  %tmp20 = load i32, ptr %tmp19, align 4
  store i32 4, ptr %tmp19
  br label %bb21

bb21:
  %tmp23 = add nuw nsw i64 %tmp12, 1
  br i1 true, label %bb24, label %bb25

bb24:
  br label %bb11

bb25:
  ret void
}
