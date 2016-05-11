; RUN: opt -instcombine -S < %s | FileCheck %s

; CHECK-LABEL: @test1
; CHECK: %cmp1 = icmp ugt i32 %d, %n
define i1 @test1(i32 %n, i32 %d) {
  %div = udiv i32 %n, %d
  %cmp1 = icmp eq i32 %div, 0
  ret i1 %cmp1
}

; CHECK-LABEL: @test2
; CHECK: %cmp1 = icmp ugt i32 %d, 64
define i1 @test2(i32 %d) {
  %div = udiv i32 64, %d
  %cmp1 = icmp eq i32 %div, 0
  ret i1 %cmp1
}

; CHECK-LABEL: @test3
; CHECK: %cmp1 = icmp ule i32 %d, %n
define i1 @test3(i32 %n, i32 %d) {
  %div = udiv i32 %n, %d
  %cmp1 = icmp ne i32 %div, 0
  ret i1 %cmp1
}

; CHECK-LABEL: @test4
; CHECK: %cmp1 = icmp ult i32 %d, 65
define i1 @test4(i32 %d) {
  %div = udiv i32 64, %d
  %cmp1 = icmp ne i32 %div, 0
  ret i1 %cmp1
}

; CHECK-LABEL: @test5
; CHECK: ret i1 true
define i1 @test5(i32 %d) {
  %div = udiv i32 -1, %d
  %cmp1 = icmp ne i32 %div, 0
  ret i1 %cmp1
}

; CHECK-LABEL: @test6
; CHECK: %cmp1 = icmp ult i32 %d, 6
define i1 @test6(i32 %d) {
  %div = udiv i32 5, %d
  %cmp1 = icmp ugt i32 %div, 0
  ret i1 %cmp1
}

; (icmp ugt (udiv C1, X), C1) -> false.
; CHECK-LABEL: @test7
; CHECK: ret i1 false
define i1 @test7(i32 %d) {
  %div = udiv i32 8, %d
  %cmp1 = icmp ugt i32 %div, 8
  ret i1 %cmp1
}

; CHECK-LABEL: @test8
; CHECK: %cmp1 = icmp ult i32 %d, 2
define i1 @test8(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ugt i32 %div, 3
  ret i1 %cmp1
}

; CHECK-LABEL: @test9
; CHECK: %cmp1 = icmp ult i32 %d, 2
define i1 @test9(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ugt i32 %div, 2
  ret i1 %cmp1
}

; CHECK-LABEL: @test10
; CHECK: %cmp1 = icmp ult i32 %d, 3
define i1 @test10(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ugt i32 %div, 1
  ret i1 %cmp1
}

; CHECK-LABEL: @test11
; CHECK: %cmp1 = icmp ugt i32 %d, 4
define i1 @test11(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ult i32 %div, 1
  ret i1 %cmp1
}

; CHECK-LABEL: @test12
; CHECK: %cmp1 = icmp ugt i32 %d, 2
define i1 @test12(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ult i32 %div, 2
  ret i1 %cmp1
}

; CHECK-LABEL: @test13
; CHECK: %cmp1 = icmp ugt i32 %d, 1
define i1 @test13(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ult i32 %div, 3
  ret i1 %cmp1
}

; CHECK-LABEL: @test14
; CHECK: %cmp1 = icmp ugt i32 %d, 1
define i1 @test14(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ult i32 %div, 4
  ret i1 %cmp1
}

; icmp ugt X, UINT_MAX -> false.
; CHECK-LABEL: @test15
; CHECK: ret i1 false
define i1 @test15(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ugt i32 %div, -1
  ret i1 %cmp1
}

; icmp ult X, UINT_MAX -> true.
; CHECK-LABEL: @test16
; CHECK: ret i1 true
define i1 @test16(i32 %d) {
  %div = udiv i32 4, %d
  %cmp1 = icmp ult i32 %div, -1
  ret i1 %cmp1
}
