; RUN: llc -mtriple=x86_64-- -print-after=finalize-isel %s -o /dev/null 2>&1 | FileCheck %s

declare void @foo(i32)

; CHECK-LABEL: test

define void @test(i32 %x) nounwind {
entry:
  switch i32 %x, label %sw.default [
    i32 1, label %sw.bb
    i32 155, label %sw.bb
    i32 156, label %sw.bb
    i32 157, label %sw.bb
    i32 158, label %sw.bb
    i32 159, label %sw.bb
    i32 1134, label %sw.bb
    i32 1140, label %sw.bb
  ], !prof !1

sw.bb:
  call void @foo(i32 0)
  br label %sw.epilog

sw.default:
  call void @foo(i32 1)
  br label %sw.epilog

sw.epilog:
  ret void

; Check if weights are correctly assigned to edges generated from switch
; statement.
;
; CHECK: bb.0{{[0-9a-zA-Z.]*}}:
; %bb.0 to %bb.4: [0, 1133] (65 = 60 + 5)
; %bb.0 to %bb.5: [1134, UINT32_MAX] (25 = 20 + 5)
; CHECK: successors: %bb.4(0x5c71c71c), %bb.5(0x238e38e4)
;
; CHECK: bb.4{{[0-9a-zA-Z.]*}}:
; %bb.4 to %bb.1: [155, 159] (50)
; %bb.4 to %bb.5: [0, 1133] - [155, 159] (15 = 10 + 5)
; CHECK: successors: %bb.1(0x62762762), %bb.7(0x1d89d89e)
;
; CHECK: bb.5{{[0-9a-zA-Z.]*}}:
; %bb.5 to %bb.1: {1140} (10)
; %bb.5 to %bb.6: [1134, UINT32_MAX] - {1140} (15 = 10 + 5)
; CHECK: successors: %bb.1(0x33333333), %bb.6(0x4ccccccd)
;
; CHECK: bb.6{{[0-9a-zA-Z.]*}}:
; %bb.6 to %bb.1: {1134} (10)
; %bb.6 to %bb.2: [1134, UINT32_MAX] - {1134, 1140} (5)
; CHECK: successors: %bb.1(0x55555555), %bb.2(0x2aaaaaab)
}

; CHECK-LABEL: test2

define void @test2(i32 %x) nounwind {
entry:

; In this switch statement, there is an edge from jump table to default
; statement.

  switch i32 %x, label %sw.default [
    i32 1, label %sw.bb
    i32 10, label %sw.bb2
    i32 11, label %sw.bb3
    i32 12, label %sw.bb4
    i32 13, label %sw.bb5
    i32 14, label %sw.bb5
  ], !prof !3

sw.bb:
  call void @foo(i32 0)
  br label %sw.epilog

sw.bb2:
  call void @foo(i32 2)
  br label %sw.epilog

sw.bb3:
  call void @foo(i32 3)
  br label %sw.epilog

sw.bb4:
  call void @foo(i32 4)
  br label %sw.epilog

sw.bb5:
  call void @foo(i32 5)
  br label %sw.epilog

sw.default:
  call void @foo(i32 1)
  br label %sw.epilog

sw.epilog:
  ret void

; Check if weights are correctly assigned to edges generated from switch
; statement.
;
; CHECK: bb.0{{[0-9a-zA-Z.]*}}:
; %bb.0 to %bb.6: {0} + [15, UINT32_MAX] (5)
; %bb.0 to %bb.8: [1, 14] (jump table) (65 = 60 + 5)
; CHECK: successors: %bb.6(0x09249249), %bb.8(0x76db6db7)
;
; CHECK: bb.8{{[0-9a-zA-Z.]*}}:
; %bb.8 to %bb.1: {1} (10)
; %bb.8 to %bb.6: [2, 9] (5)
; %bb.8 to %bb.2: {10} (10)
; %bb.8 to %bb.3: {11} (10)
; %bb.8 to %bb.4: {12} (10)
; %bb.8 to %bb.5: {13, 14} (20)
; CHECK: successors: %bb.1(0x13b13b14), %bb.6(0x09d89d8a), %bb.2(0x13b13b14), %bb.3(0x13b13b14), %bb.4(0x13b13b14), %bb.5(0x27627628)
}

; CHECK-LABEL: test3

define void @test3(i32 %x) nounwind {
entry:

; In this switch statement, there is no edge from jump table to default
; statement.

  switch i32 %x, label %sw.default [
    i32 10, label %sw.bb
    i32 11, label %sw.bb2
    i32 12, label %sw.bb3
    i32 13, label %sw.bb4
    i32 14, label %sw.bb5
  ], !prof !2

sw.bb:
  call void @foo(i32 0)
  br label %sw.epilog

sw.bb2:
  call void @foo(i32 2)
  br label %sw.epilog

sw.bb3:
  call void @foo(i32 3)
  br label %sw.epilog

sw.bb4:
  call void @foo(i32 4)
  br label %sw.epilog

sw.bb5:
  call void @foo(i32 5)
  br label %sw.epilog

sw.default:
  call void @foo(i32 1)
  br label %sw.epilog

sw.epilog:
  ret void

; Check if weights are correctly assigned to edges generated from switch
; statement.
;
; CHECK: bb.0{{[0-9a-zA-Z.]*}}:
; %bb.0 to %bb.6: [0, 9] + [15, UINT32_MAX] {10}
; %bb.0 to %bb.8: [10, 14] (jump table) (50)
; CHECK: successors: %bb.6(0x15555555), %bb.8(0x6aaaaaab)
;
; CHECK: bb.8{{[0-9a-zA-Z.]*}}:
; %bb.8 to %bb.1: {10} (10)
; %bb.8 to %bb.2: {11} (10)
; %bb.8 to %bb.3: {12} (10)
; %bb.8 to %bb.4: {13} (10)
; %bb.8 to %bb.5: {14} (10)
; CHECK: successors: %bb.1(0x1999999a), %bb.2(0x1999999a), %bb.3(0x1999999a), %bb.4(0x1999999a), %bb.5(0x1999999a)
}

; CHECK-LABEL: test4

define void @test4(i32 %x) nounwind {
entry:

; In this switch statement, there is no edge from bit test to default basic
; block.

  switch i32 %x, label %sw.default [
    i32 1, label %sw.bb
    i32 111, label %sw.bb2
    i32 112, label %sw.bb3
    i32 113, label %sw.bb3
    i32 114, label %sw.bb2
    i32 115, label %sw.bb2
  ], !prof !3

sw.bb:
  call void @foo(i32 0)
  br label %sw.epilog

sw.bb2:
  call void @foo(i32 2)
  br label %sw.epilog

sw.bb3:
  call void @foo(i32 3)
  br label %sw.epilog

sw.default:
  call void @foo(i32 1)
  br label %sw.epilog

sw.epilog:
  ret void

; Check if weights are correctly assigned to edges generated from switch
; statement.
;
; CHECK: bb.0{{[0-9a-zA-Z.]*}}:
; %bb.0 to %bb.6: [0, 110] + [116, UINT32_MAX] (20)
; %bb.0 to %bb.7: [111, 115] (bit test) (50)
; CHECK: successors: %bb.6(0x24924925), %bb.7(0x5b6db6db)
;
; CHECK: bb.7{{[0-9a-zA-Z.]*}}:
; %bb.7 to %bb.2: {111, 114, 115} (30)
; %bb.7 to %bb.3: {112, 113} (20)
; CHECK: successors: %bb.2(0x4ccccccd), %bb.3(0x33333333)
}

; CHECK-LABEL: test5

define void @test5(i32 %x) nounwind {
entry:

; In this switch statement, there is an edge from jump table to default basic
; block.

  switch i32 %x, label %sw.default [
    i32 4, label %sw.bb
    i32 20, label %sw.bb2
    i32 28, label %sw.bb3
    i32 36, label %sw.bb4
    i32 124, label %sw.bb5
  ], !prof !2

sw.bb:
  call void @foo(i32 0)
  br label %sw.epilog

sw.bb2:
  call void @foo(i32 1)
  br label %sw.epilog

sw.bb3:
  call void @foo(i32 2)
  br label %sw.epilog

sw.bb4:
  call void @foo(i32 3)
  br label %sw.epilog

sw.bb5:
  call void @foo(i32 4)
  br label %sw.epilog

sw.default:
  call void @foo(i32 5)
  br label %sw.epilog

sw.epilog:
  ret void

; Check if weights are correctly assigned to edges generated from switch
; statement.
;
; CHECK: bb.0{{[0-9a-zA-Z.]*}}:
; %bb.0 to %bb.6: [10, UINT32_MAX] (15)
; %bb.0 to %bb.8: [4, 20, 28, 36] (jump table) (45)
; CHECK: successors: %bb.8(0x20000001), %bb.9(0x5fffffff)
}

!1 = !{!"branch_weights", i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10}
!2 = !{!"branch_weights", i32 10, i32 10, i32 10, i32 10, i32 10, i32 10}
!3 = !{!"branch_weights", i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10}
