; RUN: opt < %s -gvn -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

; rdar://12801584
; Value of %shouldExit can be changed by RunInMode.
; Make sure we do not replace load %shouldExit in while.cond.backedge
; with a phi node where the value from while.body is 0.
define i32 @test() nounwind ssp {
entry:
; CHECK: test()
; CHECK: while.body:
; CHECK: call void @RunInMode
; CHECK: br i1 %tobool, label %while.cond.backedge, label %if.then
; CHECK: while.cond.backedge:
; CHECK: load i32, i32* %shouldExit
; CHECK: br i1 %cmp, label %while.body
  %shouldExit = alloca i32, align 4
  %tasksIdle = alloca i32, align 4
  store i32 0, i32* %shouldExit, align 4
  store i32 0, i32* %tasksIdle, align 4
  call void @CTestInitialize(i32* %tasksIdle) nounwind
  %0 = load i32, i32* %shouldExit, align 4
  %cmp1 = icmp eq i32 %0, 0
  br i1 %cmp1, label %while.body.lr.ph, label %while.end

while.body.lr.ph:
  br label %while.body

while.body:
  call void @RunInMode(i32 100) nounwind
  %1 = load i32, i32* %tasksIdle, align 4
  %tobool = icmp eq i32 %1, 0
  br i1 %tobool, label %while.cond.backedge, label %if.then

if.then:
  store i32 0, i32* %tasksIdle, align 4
  call void @TimerCreate(i32* %shouldExit) nounwind
  br label %while.cond.backedge

while.cond.backedge:
  %2 = load i32, i32* %shouldExit, align 4
  %cmp = icmp eq i32 %2, 0
  br i1 %cmp, label %while.body, label %while.cond.while.end_crit_edge

while.cond.while.end_crit_edge:
  br label %while.end

while.end:
  ret i32 0
}
declare void @CTestInitialize(i32*)
declare void @RunInMode(i32)
declare void @TimerCreate(i32*)
