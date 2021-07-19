; RUN: opt -passes='simple-loop-unswitch<no-trivial>' -S < %s | FileCheck %s --check-prefix=NOTRIVIAL
; RUN: opt -passes='simple-loop-unswitch' -S < %s | FileCheck %s --check-prefix=TRIVIAL
; RUN: opt -passes='simple-loop-unswitch<trivial>' -S < %s | FileCheck %s --check-prefix=TRIVIAL

declare void @some_func() noreturn

; NOTRIVIAL-NOT: split
; TRIVIAL: split
define i32 @test1(i32* %var, i1 %cond1, i1 %cond2) {
entry:
  br label %loop_begin

loop_begin:
  br i1 %cond1, label %continue, label %loop_exit	; first trivial condition

continue:
  %var_val = load i32, i32* %var
  br i1 %cond2, label %do_something, label %loop_exit	; second trivial condition

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}