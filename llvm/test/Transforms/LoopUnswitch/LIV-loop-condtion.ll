; RUN: opt < %s -loop-unswitch -loop-unswitch-threshold=0 -S 2>&1 | FileCheck %s
; RUN: opt < %s -loop-unswitch -loop-unswitch-threshold=0 -enable-mssa-loop-dependency=true -verify-memoryssa -S 2>&1 | FileCheck %s

; This is to test trivial loop unswitch only happens when trivial condition
; itself is an LIV loop condition (not partial LIV which could occur in and/or).

define i32 @test(i1 %cond1, i32 %var1) {
entry:
  br label %loop_begin

loop_begin:
  %var3 = phi i32 [%var1, %entry], [%var2, %do_something]
  %cond2 = icmp eq i32 %var3, 10
  %cond.and = and i1 %cond1, %cond2
  
; %cond.and only has %cond1 as LIV so no unswitch should happen.
; CHECK: br i1 %cond.and, label %do_something, label %loop_exit
  br i1 %cond.and, label %do_something, label %loop_exit 

do_something:
  %var2 = add i32 %var3, 1
  call void @some_func() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}

declare void @some_func() noreturn 
