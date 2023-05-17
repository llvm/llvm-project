; RUN: opt < %s -S -passes="default<O1>" | FileCheck %s -check-prefixes=TRIVIAL,CHECK
; RUN: opt < %s -S -passes="default<O2>" | FileCheck %s -check-prefixes=TRIVIAL,CHECK
; RUN: opt < %s -S -passes="default<O3>" | FileCheck %s -check-prefixes=NONTRIVIAL,CHECK
; RUN: opt < %s -S -passes="default<Os>" | FileCheck %s -check-prefixes=TRIVIAL,CHECK
; RUN: opt < %s -S -passes="default<Oz>" | FileCheck %s -check-prefixes=TRIVIAL,CHECK

declare i32 @a()
declare i32 @b()
declare i32 @c()

; TRIVIAL-NOT: loop_begin.us:
; NONTRIVIAL: loop_begin.us:

define i32 @test1(ptr %ptr, i1 %cond1, i1 %cond2) {
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
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret i32 0
}

; CHECK-NOT: loop2_begin.us:
define i32 @test2(ptr %ptr, i1 %cond1, i1 %cond2) optsize {
entry:
  br label %loop2_begin

loop2_begin:
  br i1 %cond1, label %loop2_a, label %loop2_b

loop2_a:
  call i32 @a()
  br label %latch2

loop2_b:
  br i1 %cond2, label %loop2_b_a, label %loop2_b_b

loop2_b_a:
  call i32 @b()
  br label %latch2

loop2_b_b:
  call i32 @c()
  br label %latch2

latch2:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop2_begin, label %loop2_exit

loop2_exit:
  ret i32 0
}
