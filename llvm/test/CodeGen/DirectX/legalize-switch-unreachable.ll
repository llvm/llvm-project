; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Test that a switch with an unreachable default and a common successor across
; all case blocks has its default redirected to the common successor.

define i32 @test_common_successor(i32 %val) {
; CHECK-LABEL: define i32 @test_common_successor(
; CHECK:       entry:
; CHECK-NEXT:    switch i32 %val, label %merge [
; CHECK-NEXT:      i32 0, label %case0
; CHECK-NEXT:      i32 1, label %case1
; CHECK-NEXT:      i32 2, label %case2
; CHECK-NEXT:    ]
; CHECK:       case0:
; CHECK-NEXT:    br label %merge
; CHECK:       case1:
; CHECK-NEXT:    br label %merge
; CHECK:       case2:
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    %result = phi i32 [ 10, %case0 ], [ 20, %case1 ], [ 30, %case2 ], [ poison, %entry ]
; CHECK-NEXT:    ret i32 %result
;
entry:
  switch i32 %val, label %default [
    i32 0, label %case0
    i32 1, label %case1
    i32 2, label %case2
  ]

default:
  unreachable

case0:
  br label %merge

case1:
  br label %merge

case2:
  br label %merge

merge:
  %result = phi i32 [ 10, %case0 ], [ 20, %case1 ], [ 30, %case2 ]
  ret i32 %result
}

; Test that a switch with an unreachable default and no common successor
; has its default redirected to the first case block.

define i32 @test_no_common_successor(i32 %val) {
; CHECK-LABEL: define i32 @test_no_common_successor(
; CHECK:       entry:
; CHECK-NEXT:    switch i32 %val, label %case0 [
; CHECK-NEXT:      i32 0, label %case0
; CHECK-NEXT:      i32 1, label %case1
; CHECK-NEXT:    ]
; CHECK:       case0:
; CHECK-NEXT:    ret i32 10
; CHECK:       case1:
; CHECK-NEXT:    ret i32 20
;
entry:
  switch i32 %val, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]

default:
  unreachable

case0:
  ret i32 10

case1:
  ret i32 20
}

; Test that a switch with a reachable default is not modified.

define i32 @test_reachable_default(i32 %val) {
; CHECK-LABEL: define i32 @test_reachable_default(
; CHECK:       entry:
; CHECK-NEXT:    switch i32 %val, label %default [
; CHECK-NEXT:      i32 0, label %case0
; CHECK-NEXT:    ]
; CHECK:       default:
; CHECK-NEXT:    ret i32 -1
; CHECK:       case0:
; CHECK-NEXT:    ret i32 10
;
entry:
  switch i32 %val, label %default [
    i32 0, label %case0
  ]

default:
  ret i32 -1

case0:
  ret i32 10
}

; Test with conditional branches in case blocks (no common successor) and
; the default falling back to first case.

define i32 @test_conditional_case_blocks(i32 %val, i1 %cond) {
; CHECK-LABEL: define i32 @test_conditional_case_blocks(
; CHECK:       entry:
; CHECK-NEXT:    switch i32 %val, label %case0 [
; CHECK-NEXT:      i32 0, label %case0
; CHECK-NEXT:      i32 1, label %case1
; CHECK-NEXT:    ]
; CHECK:       case0:
; CHECK-NEXT:    br i1 %cond, label %merge_a, label %merge_b
; CHECK:       case1:
; CHECK-NEXT:    br label %merge_a
; CHECK:       merge_a:
; CHECK-NEXT:    ret i32 1
; CHECK:       merge_b:
; CHECK-NEXT:    ret i32 2
;
entry:
  switch i32 %val, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]

default:
  unreachable

case0:
  br i1 %cond, label %merge_a, label %merge_b

case1:
  br label %merge_a

merge_a:
  ret i32 1

merge_b:
  ret i32 2
}
