; RUN: opt -passes='loop(loop-instsimplify,loop-simplifycfg,unswitch),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -verify-memoryssa -passes='loop-mssa(loop-instsimplify,loop-simplifycfg,unswitch),verify<loops>' -S < %s | FileCheck %s

declare void @some_func() noreturn

define i32 @test1(i32* %var, i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test1(
entry:
  br label %loop_begin
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %{{.*}}, label %entry.split, label %loop_exit.split
;
; CHECK:       entry.split:
; CHECK-NEXT:    br i1 %{{.*}}, label %entry.split.split, label %loop_exit
;
; CHECK:       entry.split.split:
; CHECK-NEXT:    br label %loop_begin

loop_begin:
  br i1 %cond1, label %continue, label %loop_exit ; first trivial condition

continue:
  %var_val = load i32, i32* %var
  %var_cond = trunc i32 %var_val to i1
  %maybe_cond = select i1 %cond1, i1 %cond2, i1 %var_cond
  br i1 %maybe_cond, label %do_something, label %loop_exit ; second trivial condition

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin
; CHECK:       loop_begin:
; CHECK-NEXT:    call
; CHECK-NEXT:    br label %loop_begin

loop_exit:
  ret i32 0
; CHECK:       loop_exit:
; CHECK-NEXT:    br label %loop_exit.split
;
; CHECK:       loop_exit.split:
; CHECK-NEXT:    ret
}
