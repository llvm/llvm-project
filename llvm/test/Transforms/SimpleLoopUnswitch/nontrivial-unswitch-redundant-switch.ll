; REQUIRES: asserts
; RUN: opt -passes='unswitch<nontrivial>' -disable-output -S < %s
; RUN: opt -passes='loop-mssa(unswitch<nontrivial>)' -disable-output -S < %s
; RUN: opt -simple-loop-unswitch -enable-nontrivial-unswitch -disable-output -S < %s

; This loop shouldn't trigger asserts in SimpleLoopUnswitch.
define void @test_redundant_switch(i1* %ptr, i32 %cond) {
entry:
  br label %loop_begin

loop_begin:
  switch i32 %cond, label %loop_body [
      i32 0, label %loop_body
  ]

loop_body:
  br label %loop_latch

loop_latch:
  %v = load i1, i1* %ptr
  br i1 %v, label %loop_begin, label %loop_exit

loop_exit:
  ret void
}
