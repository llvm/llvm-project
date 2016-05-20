; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s

define void @test1(i32 %c) {
; CHECK-LABEL: test1:
entry:
  %0 = alloca i8, i32 %c
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.end, label %if.then

if.end:
  call void @g(i8* %0)
  ret void

if.then:
  call void @crash(i8* %0)
  unreachable
; CHECK: calll _crash
; There is no need to adjust the stack after the call, since
; the function is noreturn and that code will therefore never run.
; CHECK-NOT: add
; CHECK-NOT: pop
}

define void @test2(i32 %c) {
; CHECK-LABEL: test2:
entry:
  %0 = alloca i8, i32 %c
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.end, label %if.then

if.end:
  call void @g(i8* %0)
  ret void

if.then:
  call void @crash2(i8* %0)
  unreachable
; CHECK: calll _crash2
; Even though _crash2 is not marked noreturn, it is in practice because
; of the "unreachable" right after it. This happens e.g. when falling off
; a non-void function after a call.
; CHECK-NOT: add
; CHECK-NOT: pop
}

declare void @crash(i8*) noreturn
declare void @crash2(i8*)
declare void @g(i8*)
