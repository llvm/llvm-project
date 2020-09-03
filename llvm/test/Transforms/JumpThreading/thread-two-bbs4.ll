; RUN: opt < %s -jump-threading -S -verify | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Verify that we do *not* thread any edge.  We used to evaluate
; constant expressions like:
;
;   icmp ugt i8* null, inttoptr (i64 4 to i8*)
;
; as "true", causing jump threading to a wrong destination.
define void @foo(i8* %arg1, i8* %arg2) {
; CHECK-LABEL: @foo
; CHECK-NOT: bb_{{[^ ]*}}.thread:
entry:
  %cmp1 = icmp eq i8* %arg1, null
  br i1 %cmp1, label %bb_bar1, label %bb_end

bb_bar1:
  call void @bar(i32 1)
  br label %bb_end

bb_end:
  %cmp2 = icmp ne i8* %arg2, null
  br i1 %cmp2, label %bb_cont, label %bb_bar2

bb_bar2:
  call void @bar(i32 2)
  br label %bb_exit

bb_cont:
  %cmp3 = icmp ule i8* %arg1, inttoptr (i64 4 to i8*)
  br i1 %cmp3, label %bb_exit, label %bb_bar3

bb_bar3:
  call void @bar(i32 3)
  br label %bb_exit

bb_exit:
  ret void
}

declare void @bar(i32)
