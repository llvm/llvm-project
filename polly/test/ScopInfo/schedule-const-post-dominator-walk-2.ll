; RUN: opt %loadPolly -analyze -polly-scops < %s | FileCheck %s

; The SCoP contains a loop with multiple exit blocks (BBs after leaving
; the loop). The current implementation of deriving their domain derives
; only a common domain for all of the exit blocks. We disabled loops with
; multiple exit blocks until this is fixed.
; XFAIL: *

; CHECK: Stmt_loopA[i0] -> [0, 0, 0]
; CHECK-DAG: Stmt_loopB[i0] -> [0, 0, 1]
; CHECK-DAG: Stmt_bbB[] -> [1, 0, 0]
; CHECK-DAG: Stmt_bbA[] -> [2, 0, 0]
; CHECK-DAG: Stmt_bbMerge[]   -> [3, 0, 0]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @hoge(i64 %p0, i64 %p1, i64 %p2, i64 %p3, float* %A) {
entry:
  br label %loopA

loopA:
  %tmp4 = phi i64 [ 0, %entry ], [ 0, %loopB]
  store float 42.0, float* %A
  %cmp0 = icmp sle i64 %p0, 100
  br i1 %cmp0, label %loopB, label %bbB

loopB:
  store float 42.0, float* %A
  %cmp1 = icmp sle i64 %p1, 100
  br i1 %cmp1, label %loopA, label %bbA

bbA:
  store float 42.0, float* %A
  %cmpbbA = icmp sle i64 %p2, 50
  br i1 %cmpbbA, label %bbMerge, label %exit

bbB:
  store float 42.0, float* %A
  %cmpbbB= icmp sle i64 %p3, 200
  br i1 %cmpbbB, label %exit, label %bbMerge

bbMerge:
  store float 42.0, float* %A
  br label %exit

exit:
  ret void
}
