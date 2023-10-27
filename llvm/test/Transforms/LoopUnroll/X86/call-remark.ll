; RUN: opt -passes=debugify,loop-unroll -mcpu=znver3 -pass-remarks=loop-unroll -pass-remarks-analysis=loop-unroll < %s -S 2>&1 | FileCheck --check-prefixes=ALL,UNROLL %s
; RUN: opt -passes=debugify,loop-unroll -mcpu=znver3 -pass-remarks=TTI -pass-remarks-analysis=TTI  < %s -S 2>&1 | FileCheck --check-prefixes=ALL,TTI %s
; RUN: opt -passes=debugify,loop-unroll -mcpu=znver4 -pass-remarks=loop-unroll -pass-remarks-analysis=loop-unroll < %s -S 2>&1 | FileCheck --check-prefixes=ALL,UNROLL %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; TTI: remark: <stdin>:7:1: advising against unrolling the loop because it contains a call
; UNROLL: remark: <stdin>:14:1: unrolled loop by a factor of 8 with run-time trip count

define void @contains_external_call(i32 %count) {
; ALL-LABEL: @contains_external_call(
; ALL-NOT: unroll
entry:
  %cmp.not3 = icmp eq i32 %count, 0
  br i1 %cmp.not3, label %for.cond.cleanup, label %for.body

for.body:
  %i.04 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  tail call void @sideeffect()
  %inc = add nuw nsw i32 %i.04, 1
  %cmp.not = icmp eq i32 %inc, %count
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

declare void @sideeffect()

define i32 @no_external_calls(i32 %count) {
; ALL-LABEL: @no_external_calls(
; ALL: unroll
entry:
  %cmp.not5 = icmp eq i32 %count, 0
  br i1 %cmp.not5, label %for.end, label %for.body

for.body:
  %i.06 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %inc = add nuw nsw i32 %i.06, 1
  %cmp.not = icmp eq i32 %inc, %count
  br i1 %cmp.not, label %for.end, label %for.body

for.end:
  ret i32 %count
}
