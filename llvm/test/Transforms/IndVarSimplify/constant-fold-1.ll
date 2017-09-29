; RUN: opt < %s -indvars -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test(i64* %arg) unnamed_addr align 2 {
bb:
  switch i64 undef, label %bb1 [
  ]

bb1:                                              ; preds = %bb
  br label %bb2

bb2:                                              ; preds = %bb6, %bb1
  %tmp = phi i64* [%arg, %bb1 ], [ %tmp7, %bb6 ]
  switch i2 undef, label %bb6 [
    i2 1, label %bb5
    i2 -2, label %bb3
    i2 -1, label %bb3
  ]

bb3:                                              ; preds = %bb2, %bb2
  %tmp4 = call fastcc i32* @wobble(i64* nonnull %tmp, i32* null)
  %tmp5 = load i32, i32* %tmp4 , align 8
  br label %bb6

bb5:                                              ; preds = %bb2
  unreachable

bb6:                                              ; preds = %bb3, %bb2
  %tmp7 = load i64*, i64** undef, align 8
  br label %bb2
}

declare i32* @wobble(i64*, i32* returned)

; Should not fail when SCEV is fold to ConstantPointerNull
; CHECK-LABEL: void @test
; CHECK:         load i32, i32* %{{[a-zA-Z$._0-9]+}}
