; RUN: opt < %s -simple-loop-unswitch -verify-memoryssa -S 2>&1 | FileCheck %s

; This is to test trivial loop unswitch only happens when trivial condition
; itself is an LIV loop condition (not partial LIV which could occur in and/or).

define i32 @test(i1 %cond1, i32 %var1) {
; CHECK-LABEL: define i32 @test(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FROZEN:%.+]] = freeze i1 %cond1
; CHECK-NEXT:    br i1 [[FROZEN]], label %entry.split, label %loop_exit.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  %var3 = phi i32 [%var1, %entry], [%var2, %do_something]
  %cond2 = icmp eq i32 %var3, 10
  %cond.and = and i1 %cond1, %cond2
  br i1 %cond.and, label %do_something, label %loop_exit
; CHECK:       loop_begin:
; CHECK-NEXT:    %[[VAR3:.*]] = phi i32
; CHECK-NEXT:    %[[COND2:.*]] = icmp eq i32 %[[VAR3]], 10
; CHECK-NEXT:    %[[COND_AND:.*]] = and i1 true, %[[COND2]]
; CHECK-NEXT:    br i1 %[[COND_AND]], label %do_something, label %loop_exit

do_something:
  %var2 = add i32 %var3, 1
  call void @some_func() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}

declare void @some_func() noreturn 
