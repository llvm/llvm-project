; Test case for crash with Floating point Exception in loop-vectorize pass
; This test verifies that the loop vectorizer does not crash with SIGFPE
; when processing blocks with zero block frequency.
; See issue #172049

; RUN: opt -passes=loop-vectorize -S %s

; ModuleID = 'reduced.ll'
source_filename = "reduced.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define ptr addrspace(1) @wombat() gc "statepoint-example" {
bb:
  br label %bb2

bb1:
  ret ptr addrspace(1) null

bb2:
  %phi = phi i64 [ %add, %bb6 ], [ 0, %bb ]
  br i1 false, label %bb3, label %bb6

bb3:
  br i1 false, label %bb4, label %bb5, !prof !0

bb4:
  br label %bb6

bb5:
  br label %bb6

bb6:
  %add = add i64 %phi, 1
  %icmp = icmp eq i64 %phi, 0
  br i1 %icmp, label %bb2, label %bb1
}

!0 = !{!"branch_weights", i32 1, i32 0}
