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
