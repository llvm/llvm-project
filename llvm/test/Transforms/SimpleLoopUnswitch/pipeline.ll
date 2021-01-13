; RUN: opt < %s -S -passes="default<O1>" | FileCheck %s -check-prefix=O1
; RUN: opt < %s -S -passes="default<O2>" | FileCheck %s -check-prefix=O2

declare i32 @a()
declare i32 @b()
declare i32 @c()

; O1-NOT: loop_begin.us:
; O2: loop_begin.us:

define i32 @test1(i1* %ptr, i1 %cond1, i1 %cond2) {
entry:
  br label %loop_begin

loop_begin:
  br i1 %cond1, label %loop_a, label %loop_b

loop_a:
  call i32 @a()
  br label %latch

loop_b:
  br i1 %cond2, label %loop_b_a, label %loop_b_b

loop_b_a:
  call i32 @b()
  br label %latch

loop_b_b:
  call i32 @c()
  br label %latch

latch:
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret i32 0
}
