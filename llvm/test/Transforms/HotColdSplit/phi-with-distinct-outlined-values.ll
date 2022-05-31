; RUN: opt -S -hotcoldsplit -hotcoldsplit-threshold=0 < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo(
; CHECK: phi i32 [ 0, %entry ], [ %p.ce.reload, %codeRepl ]

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK: call {{.*}}@sink
; CHECK: %p.ce = phi i32 [ 1, %coldbb ], [ 3, %coldbb2 ]
; CHECK-NEXT: store i32 %p.ce, ptr %p.ce.out 

define void @foo(i32 %cond) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.end, label %coldbb

coldbb:
  call void @sink()
  call void @sideeffect()
  br i1 undef, label %if.end, label %coldbb2

coldbb2:
  br label %if.end

if.end:
  %p = phi i32 [0, %entry], [1, %coldbb], [3, %coldbb2]
  ret void
}

declare void @sink() cold

declare void @sideeffect()
