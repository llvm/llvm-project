; RUN: opt -S -passes=loop-reduce -scalar-evolution-max-arith-depth=0 %s | FileCheck %s
;
; Make sure we don't trigger an assertion in SCEV here.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @test(i32 %phi) {
; CHECK-LABEL: test
bb:
  br label %bb6

bb6:                                              ; preds = %bb6, %bb
  %phi7 = phi i32 [ 1, %bb ], [ %add44, %bb6 ]
  %mul13 = mul i32 %phi7, %phi
  %mul16 = mul i32 %mul13, 0
  %add44 = add i32 %phi7, 1
  br i1 true, label %bb51, label %bb6

bb51:                                             ; preds = %bb6
  unreachable
}

