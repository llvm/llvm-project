; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -verify-memoryssa -S < %s 2>&1 | FileCheck %s

; This test checks if unswitched condition preserve make.implicit metadata.

define i32 @test(i1 %cond) {
; CHECK-LABEL: @test(
; CHECK:  br i1 %cond, label %..split_crit_edge, label %.loop_exit.split_crit_edge, !make.implicit !0
  br label %loop_begin

loop_begin:
  br i1 %cond, label %continue, label %loop_exit, !make.implicit !0

continue:
  call void @some_func()
  br label %loop_begin

loop_exit:
  ret i32 0
}

declare void @some_func()

!0 = !{}
