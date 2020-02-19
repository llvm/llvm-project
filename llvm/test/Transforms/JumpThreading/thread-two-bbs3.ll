; RUN: opt < %s -jump-threading -S -verify | FileCheck %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.16.27026"

; Verify that we do *not* thread any edge.  On Windows, we used to
; improperly duplicate EH pads like bb_cleanup below, resulting in an
; assertion failure later down the pass pipeline.
define void @foo([2 x i8]* %0) personality i8* bitcast (i32 ()* @baz to i8*) {
; CHECK-LABEL: @foo
; CHECK-NOT: bb_{{[^ ]*}}.thread:
entry:
  invoke void @bar()
          to label %bb_invoke unwind label %bb_cleanuppad

bb_invoke:
  invoke void @bar()
          to label %bb_exit unwind label %bb_cleanuppad

bb_cleanuppad:
  %index = phi i64 [ 1, %bb_invoke ], [ 0, %entry ]
  %cond1 = phi i1 [ false, %bb_invoke ], [ true, %entry ]
  %1 = cleanuppad within none []
  br i1 %cond1, label %bb_action, label %bb_cleanupret

bb_action:
  %cond2 = icmp eq i64 %index, 0
  br i1 %cond2, label %bb_cleanupret, label %bb_exit

bb_exit:
  call void @bar()
  ret void

bb_cleanupret:
  cleanupret from %1 unwind to caller
}

declare void @bar()
declare i32 @baz()
