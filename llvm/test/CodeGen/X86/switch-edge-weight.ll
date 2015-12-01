; RUN: llc -march=x86-64 -print-machineinstrs=expand-isel-pseudos %s -o /dev/null 2>&1 | FileCheck %s

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
; CHECK: BB#0:
; BB#0 to BB#4: [0, 1133] (65 = 60 + 5)
; BB#0 to BB#5: [1134, UINT32_MAX] (25 = 20 + 5)
; CHECK: Successors according to CFG: BB#4({{[0-9a-fx/= ]+}}72.22%) BB#5({{[0-9a-fx/= ]+}}27.78%)
;
; CHECK: BB#4:
; BB#4 to BB#1: [155, 159] (50)
; BB#4 to BB#5: [0, 1133] - [155, 159] (15 = 10 + 5)
; CHECK: Successors according to CFG: BB#1({{[0-9a-fx/= ]+}}76.92%) BB#7({{[0-9a-fx/= ]+}}23.08%)
;
; CHECK: BB#5:
; BB#5 to BB#1: {1140} (10)
; BB#5 to BB#6: [1134, UINT32_MAX] - {1140} (15 = 10 + 5)
; CHECK: Successors according to CFG: BB#1({{[0-9a-fx/= ]+}}40.00%) BB#6({{[0-9a-fx/= ]+}}60.00%)
;
; CHECK: BB#6:
; BB#6 to BB#1: {1134} (10)
; BB#6 to BB#2: [1134, UINT32_MAX] - {1134, 1140} (5)
; CHECK: Successors according to CFG: BB#1({{[0-9a-fx/= ]+}}66.67%) BB#2({{[0-9a-fx/= ]+}}33.33%)
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
; CHECK: BB#0:
; BB#0 to BB#6: {0} + [15, UINT32_MAX] (5)
; BB#0 to BB#8: [1, 14] (jump table) (65 = 60 + 5)
; CHECK: Successors according to CFG: BB#6({{[0-9a-fx/= ]+}}7.14%) BB#8({{[0-9a-fx/= ]+}}92.86%
;
; CHECK: BB#8:
; BB#8 to BB#1: {1} (10)
; BB#8 to BB#6: [2, 9] (5)
; BB#8 to BB#2: {10} (10)
; BB#8 to BB#3: {11} (10)
; BB#8 to BB#4: {12} (10)
; BB#8 to BB#5: {13, 14} (20)
; CHECK: Successors according to CFG: BB#1({{[0-9a-fx/= ]+}}14.29%) BB#6({{[0-9a-fx/= ]+}}7.14%) BB#2({{[0-9a-fx/= ]+}}14.29%) BB#3({{[0-9a-fx/= ]+}}14.29%) BB#4({{[0-9a-fx/= ]+}}14.29%) BB#5({{[0-9a-fx/= ]+}}28.57%)
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
; CHECK: BB#0:
; BB#0 to BB#6: [0, 9] + [15, UINT32_MAX] {10}
; BB#0 to BB#8: [10, 14] (jump table) (50)
; CHECK: Successors according to CFG: BB#6({{[0-9a-fx/= ]+}}16.67%) BB#8({{[0-9a-fx/= ]+}}83.33%)
;
; CHECK: BB#8:
; BB#8 to BB#1: {10} (10)
; BB#8 to BB#2: {11} (10)
; BB#8 to BB#3: {12} (10)
; BB#8 to BB#4: {13} (10)
; BB#8 to BB#5: {14} (10)
; CHECK: Successors according to CFG: BB#1({{[0-9a-fx/= ]+}}20.00%) BB#2({{[0-9a-fx/= ]+}}20.00%) BB#3({{[0-9a-fx/= ]+}}20.00%) BB#4({{[0-9a-fx/= ]+}}20.00%) BB#5({{[0-9a-fx/= ]+}}20.00%)
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
; CHECK: BB#0:
; BB#0 to BB#6: [0, 110] + [116, UINT32_MAX] (20)
; BB#0 to BB#7: [111, 115] (bit test) (50)
; CHECK: Successors according to CFG: BB#6({{[0-9a-fx/= ]+}}28.57%) BB#7({{[0-9a-fx/= ]+}}71.43%)
;
; CHECK: BB#7:
; BB#7 to BB#2: {111, 114, 115} (30)
; BB#7 to BB#3: {112, 113} (20)
; CHECK: Successors according to CFG: BB#2({{[0-9a-fx/= ]+}}60.00%) BB#3({{[0-9a-fx/= ]+}}40.00%)
}

; CHECK-LABEL: test5

define void @test5(i32 %x) nounwind {
entry:

; In this switch statement, there is an edge from jump table to default basic
; block.

  switch i32 %x, label %sw.default [
    i32 1, label %sw.bb
    i32 5, label %sw.bb2
    i32 7, label %sw.bb3
    i32 9, label %sw.bb4
    i32 31, label %sw.bb5
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
; CHECK: BB#0:
; BB#0 to BB#6: [10, UINT32_MAX] (15)
; BB#0 to BB#8: [1, 5, 7, 9] (jump table) (45)
; CHECK: Successors according to CFG: BB#8({{[0-9a-fx/= ]+}}25.00%) BB#9({{[0-9a-fx/= ]+}}75.00%)
}

!1 = !{!"branch_weights", i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10} 
!2 = !{!"branch_weights", i32 10, i32 10, i32 10, i32 10, i32 10, i32 10} 
!3 = !{!"branch_weights", i32 10, i32 10, i32 10, i32 10, i32 10, i32 10, i32 10} 
