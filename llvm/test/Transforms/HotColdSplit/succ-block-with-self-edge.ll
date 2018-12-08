; RUN: opt -S -hotcoldsplit < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@exit_block_with_same_incoming_vals
; CHECK: call {{.*}}@exit_block_with_same_incoming_vals.cold.1(
; CHECK-NOT: br i1 undef
; CHECK: phi i32 [ 0, %entry ], [ %p.ce.reload, %codeRepl ]
define void @exit_block_with_same_incoming_vals(i32 %cond) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.end, label %coldbb

coldbb:
  call void @sink()
  call void @sideeffect()
  call void @sideeffect()
  br i1 undef, label %if.end, label %coldbb2

coldbb2:
  %p2 = phi i32 [0, %coldbb], [1, %coldbb2]
  br i1 undef, label %if.end, label %coldbb2

if.end:
  %p = phi i32 [0, %entry], [1, %coldbb], [1, %coldbb2]
  ret void
}

; CHECK-LABEL: define {{.*}}@exit_block_with_distinct_incoming_vals
; CHECK: call {{.*}}@exit_block_with_distinct_incoming_vals.cold.1(
; CHECK-NOT: br i1 undef
; CHECK: phi i32 [ 0, %entry ], [ %p.ce.reload, %codeRepl ]
define void @exit_block_with_distinct_incoming_vals(i32 %cond) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.end, label %coldbb

coldbb:
  call void @sink()
  call void @sideeffect()
  call void @sideeffect()
  br i1 undef, label %if.end, label %coldbb2

coldbb2:
  %p2 = phi i32 [0, %coldbb], [1, %coldbb2]
  br i1 undef, label %if.end, label %coldbb2

if.end:
  %p = phi i32 [0, %entry], [1, %coldbb], [2, %coldbb2]
  ret void
}

declare void @sink() cold

declare void @sideeffect()
