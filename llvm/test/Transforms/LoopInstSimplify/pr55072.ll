; RUN: opt -S %s -passes=loop-instsimplify | FileCheck %s
; RUN: opt -S %s -passes='loop-mssa(loop-instsimplify)' -verify-memoryssa | FileCheck %s

; XFAIL: *
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_01() {
; CHECK-LABEL: test_01
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  br label %bb3

bb2:                                              ; preds = %bb2
  br label %bb2

bb3:                                              ; preds = %bb8, %bb1
  switch i32 undef, label %bb8 [
    i32 0, label %bb4
    i32 1, label %bb6
    i32 2, label %bb5
  ]

bb4:                                              ; preds = %bb3
  %tmp = lshr exact i32 undef, 16
  br label %bb6

bb5:                                              ; preds = %bb3
  br label %bb8

bb6:                                              ; preds = %bb4, %bb3
  br label %bb8

bb7:                                              ; No predecessors!
  ret i32 %tmp

bb8:                                              ; preds = %bb6, %bb5, %bb3
  br label %bb3
}
