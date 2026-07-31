; RUN: opt -S -passes='licm' < %s | FileCheck %s

; Regression test for llvm/llvm-project#64188.
; LICM must not promote the non-atomic store out of the loop when the
; loop contains a non-nosync call that may write accessible memory.

@a = dso_local global i32 0, align 4

define ptr @hoist_store_past_non_nosync_call(i32 %N) {
; CHECK-LABEL: @hoist_store_past_non_nosync_call(
; CHECK:       for.body:
; CHECK:         store i32 %i, ptr %p32
; CHECK:         call void @may_sync
entry:
  %p = call noalias ptr @malloc(i64 4)
  %p32 = bitcast ptr %p to ptr
  %cmp = icmp slt i32 0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %i = phi i32 [ 1, %entry ], [ %inc, %for.body ]
  store i32 %i, ptr %p32, align 4
  call void @may_sync(ptr @a, i32 %i)
  %inc = add i32 %i, 1
  %cmp2 = icmp sle i32 %inc, %N
  br i1 %cmp2, label %for.body, label %for.end

for.end:
  store i32 %N, ptr %p32, align 4
  ret ptr %p
}

declare noalias ptr @malloc(i64)
declare void @may_sync(ptr, i32)
