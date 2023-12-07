; REQUIRES: asserts
; RUN: opt -passes='simple-loop-unswitch<nontrivial>' -disable-output -S < %s
; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial>)' -disable-output -S < %s

; This loop shouldn't trigger asserts in SimpleLoopUnswitch.
define void @test_redundant_switch(ptr %ptr, i32 %cond) {
entry:
  br label %loop_begin

loop_begin:
  switch i32 %cond, label %loop_body [
      i32 0, label %loop_body
  ]

loop_body:
  br label %loop_latch

loop_latch:
  %v = load i1, ptr %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
}
