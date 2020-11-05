; RUN: opt -passes='loop(unswitch),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -verify-memoryssa -passes='loop-mssa(unswitch),verify<loops>' -S < %s | FileCheck %s

declare void @incf()
declare void @decf()

define i32 @test2(i32 %c) {
; CHECK-LABEL: @test2(
  br label %loop_begin

; CHECK: !prof ![[MD0:[0-9]+]]
; CHECK: loop_begin:
; CHECK: !prof ![[MD1:[0-9]+]]
loop_begin:

  switch i32 %c, label %default [
      i32 1, label %inc
      i32 2, label %dec
  ], !prof !{!"branch_weights", i64 99, i64 1, i64 2}

inc:
  call void @incf()
  br label %loop_begin

dec:
  call void @decf()
  br label %loop_begin

default:
  ret i32 0
}

; CHECK: ![[MD0]] = !{!"branch_weights", i64 99, i64 1, i64 2}
; CHECK: ![[MD1]] = !{!"branch_weights", i64 2, i64 1}
