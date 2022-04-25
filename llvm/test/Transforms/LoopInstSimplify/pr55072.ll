; RUN: opt -S %s -passes=loop-instsimplify | FileCheck %s
; RUN: opt -S %s -passes='loop-mssa(loop-instsimplify)' -verify-memoryssa | FileCheck %s

; XFAIL: *
; REQUIRES: asserts

define i32 @test_01() {
; CHECK-LABEL: test_01
bb:
  br label %loop

loop:                                              ; preds = %bb, %loop
  %tmp = lshr exact i32 undef, 16
  br label %loop

unreached:                                              ; No predecessors!
  ret i32 %tmp
}
