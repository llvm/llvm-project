; RUN: llc  -mcpu=corei7 -mtriple=x86_64-linux < %s | FileCheck %s

; The block latch should be moved before header.
;CHECK-LABEL: test1:
;CHECK:       %latch
;CHECK:       %header
;CHECK:       %false
define i32 @test1(i32* %p) {
entry:
  br label %header

header:
  %x1 = phi i64 [0, %entry], [%x2, %latch]
  %count1 = phi i32 [0, %entry], [%count4, %latch]
  %0 = ptrtoint i32* %p to i64
  %1 = add i64 %0, %x1
  %2 = inttoptr i64 %1 to i32*
  %data = load i32, i32* %2
  %3 = icmp eq i32 %data, 0
  br i1 %3, label %latch, label %false

false:
  %count2 = add i32 %count1, 1
  br label %latch

latch:
  %count4 = phi i32 [%count2, %false], [%count1, %header]
  %x2 = add i64 %x1, 1
  %4 = icmp eq i64 %x2, 100
  br i1 %4, label %exit, label %header

exit:
  ret i32 %count4
}

; The block latch and one of false/true should be moved before header.
;CHECK-LABEL: test2:
;CHECK:       %true
;CHECK:       %latch
;CHECK:       %header
;CHECK:       %false
define i32 @test2(i32* %p) {
entry:
  br label %header

header:
  %x1 = phi i64 [0, %entry], [%x2, %latch]
  %count1 = phi i32 [0, %entry], [%count4, %latch]
  %0 = ptrtoint i32* %p to i64
  %1 = add i64 %0, %x1
  %2 = inttoptr i64 %1 to i32*
  %data = load i32, i32* %2
  %3 = icmp eq i32 %data, 0
  br i1 %3, label %true, label %false

false:
  %count2 = add i32 %count1, 1
  br label %latch

true:
  %count3 = add i32 %count1, 2
  br label %latch

latch:
  %count4 = phi i32 [%count2, %false], [%count3, %true]
  %x2 = add i64 %x1, 1
  %4 = icmp eq i64 %x2, 100
  br i1 %4, label %exit, label %header

exit:
  ret i32 %count4
}

; More blocks can be moved before header.
;            header <------------
;              /\               |
;             /  \              |
;            /    \             |
;           /      \            |
;          /        \           |
;        true      false        |
;         /\         /\         |
;        /  \       /  \        |
;       /    \     /    \       |
;    true3 false3 /      \      |
;      \    /   true2  false2   |
;       \  /      \      /      |
;        \/        \    /       |
;      endif3       \  /        |
;         \          \/         |
;          \       endif2       |
;           \        /          |
;            \      /           |
;             \    /            |
;              \  /             |
;               \/              |
;              latch-------------
;                |
;                |
;              exit
;
; Blocks true3,endif3,latch should be moved before header.
;
;CHECK-LABEL: test3:
;CHECK:       %true3
;CHECK:       %endif3
;CHECK:       %latch
;CHECK:       %header
;CHECK:       %false
define i32 @test3(i32* %p) {
entry:
  br label %header

header:
  %x1 = phi i64 [0, %entry], [%x2, %latch]
  %count1 = phi i32 [0, %entry], [%count12, %latch]
  %0 = ptrtoint i32* %p to i64
  %1 = add i64 %0, %x1
  %2 = inttoptr i64 %1 to i32*
  %data = load i32, i32* %2
  %3 = icmp eq i32 %data, 0
  br i1 %3, label %true, label %false, !prof !3

false:
  %count2 = add i32 %count1, 1
  %cond = icmp sgt i32 %count2, 10
  br i1 %cond, label %true2, label %false2

false2:
  %count3 = and i32 %count2, 7
  br label %endif2

true2:
  %count4 = mul i32 %count2, 3
  br label %endif2

endif2:
  %count5 = phi i32 [%count3, %false2], [%count4, %true2]
  %count6 = sub i32 %count5, 5
  br label %latch

true:
  %count7 = add i32 %count1, 2
  %cond2 = icmp slt i32 %count7, 20
  br i1 %cond2, label %true3, label %false3

false3:
  %count8 = or i32 %count7, 3
  br label %endif3

true3:
  %count9 = xor i32 %count7, 55
  br label %endif3

endif3:
  %count10 = phi i32 [%count8, %false3], [%count9, %true3]
  %count11 = add i32 %count10, 3
  br label %latch

latch:
  %count12 = phi i32 [%count6, %endif2], [%count11, %endif3]
  %x2 = add i64 %x1, 1
  %4 = icmp eq i64 %x2, 100
  br i1 %4, label %exit, label %header

exit:
  ret i32 %count12
}

; The exit block has higher frequency than false block, so latch block
; should not moved before header.
;CHECK-LABEL: test4:
;CHECK:       %header
;CHECK:       %true
;CHECK:       %latch
;CHECK:       %false
;CHECK:       %exit
define i32 @test4(i32 %t, i32* %p) {
entry:
  br label %header

header:
  %x1 = phi i64 [0, %entry], [%x2, %latch]
  %count1 = phi i32 [0, %entry], [%count4, %latch]
  %0 = ptrtoint i32* %p to i64
  %1 = add i64 %0, %x1
  %2 = inttoptr i64 %1 to i32*
  %data = load i32, i32* %2
  %3 = icmp eq i32 %data, 0
  br i1 %3, label %true, label %false, !prof !1

false:
  %count2 = add i32 %count1, 1
  br label %latch

true:
  %count3 = add i32 %count1, 2
  br label %latch

latch:
  %count4 = phi i32 [%count2, %false], [%count3, %true]
  %x2 = add i64 %x1, 1
  %4 = icmp eq i64 %x2, 100
  br i1 %4, label %exit, label %header, !prof !2

exit:
  ret i32 %count4
}

!1 = !{!"branch_weights", i32 100, i32 1}
!2 = !{!"branch_weights", i32 16, i32 16}
!3 = !{!"branch_weights", i32 51, i32 49}

; If move latch to loop top doesn't reduce taken branch, don't do it.
;CHECK-LABEL: test5:
;CHECK:       %entry
;CHECK:       %header
;CHECK:       %latch
define void @test5(i32* %p) {
entry:
  br label %header

header:
  %x1 = phi i64 [0, %entry], [%x1, %header], [%x2, %latch]
  %0 = ptrtoint i32* %p to i64
  %1 = add i64 %0, %x1
  %2 = inttoptr i64 %1 to i32*
  %data = load i32, i32* %2
  %3 = icmp eq i32 %data, 0
  br i1 %3, label %latch, label %header

latch:
  %x2 = add i64 %x1, 1
  br label %header

exit:
  ret void
}

