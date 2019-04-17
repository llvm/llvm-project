; RUN: opt < %s -instcombine -S | FileCheck %s

; Check simplification of
; (icmp sgt x, -1) & (icmp sgt/sge n, x) --> icmp ugt/uge n, x

; CHECK-LABEL: define i1 @test_and1
; CHECK: [[R:%[0-9]+]] = icmp ugt i32 %nn, %x
; CHECK: ret i1 [[R]]
define i1 @test_and1(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp sge i32 %x, 0
  %b = icmp slt i32 %x, %nn
  %c = and i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @test_and2
; CHECK: [[R:%[0-9]+]] = icmp uge i32 %nn, %x
; CHECK: ret i1 [[R]]
define i1 @test_and2(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp sgt i32 %x, -1
  %b = icmp sle i32 %x, %nn
  %c = and i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @test_and3
; CHECK: [[R:%[0-9]+]] = icmp ugt i32 %nn, %x
; CHECK: ret i1 [[R]]
define i1 @test_and3(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp sgt i32 %nn, %x
  %b = icmp sge i32 %x, 0
  %c = and i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @test_and4
; CHECK: [[R:%[0-9]+]] = icmp uge i32 %nn, %x
; CHECK: ret i1 [[R]]
define i1 @test_and4(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp sge i32 %nn, %x
  %b = icmp sge i32 %x, 0
  %c = and i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @test_or1
; CHECK: [[R:%[0-9]+]] = icmp ule i32 %nn, %x
; CHECK: ret i1 [[R]]
define i1 @test_or1(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp slt i32 %x, 0
  %b = icmp sge i32 %x, %nn
  %c = or i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @test_or2
; CHECK: [[R:%[0-9]+]] = icmp ult i32 %nn, %x
; CHECK: ret i1 [[R]]
define i1 @test_or2(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp sle i32 %x, -1
  %b = icmp sgt i32 %x, %nn
  %c = or i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @test_or3
; CHECK: [[R:%[0-9]+]] = icmp ule i32 %nn, %x
; CHECK: ret i1 [[R]]
define i1 @test_or3(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp sle i32 %nn, %x
  %b = icmp slt i32 %x, 0
  %c = or i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @test_or4
; CHECK: [[R:%[0-9]+]] = icmp ult i32 %nn, %x
; CHECK: ret i1 [[R]]
define i1 @test_or4(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp slt i32 %nn, %x
  %b = icmp slt i32 %x, 0
  %c = or i1 %a, %b
  ret i1 %c
}

; Negative tests

; CHECK-LABEL: define i1 @negative1
; CHECK: %a = icmp
; CHECK: %b = icmp
; CHECK: %c = and i1 %a, %b
; CHECK: ret i1 %c
define i1 @negative1(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp slt i32 %x, %nn
  %b = icmp sgt i32 %x, 0      ; should be: icmp sge
  %c = and i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @negative2
; CHECK: %a = icmp
; CHECK: %b = icmp
; CHECK: %c = and i1 %a, %b
; CHECK: ret i1 %c
define i1 @negative2(i32 %x, i32 %n) {
  %a = icmp slt i32 %x, %n     ; n can be negative
  %b = icmp sge i32 %x, 0
  %c = and i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @negative3
; CHECK: %a = icmp
; CHECK: %b = icmp
; CHECK: %c = and i1 %a, %b
; CHECK: ret i1 %c
define i1 @negative3(i32 %x, i32 %y, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp slt i32 %x, %nn
  %b = icmp sge i32 %y, 0      ; should compare %x and not %y
  %c = and i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @negative4
; CHECK: %a = icmp
; CHECK: %b = icmp
; CHECK: %c = and i1 %a, %b
; CHECK: ret i1 %c
define i1 @negative4(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp ne i32 %x, %nn     ; should be: icmp slt/sle
  %b = icmp sge i32 %x, 0
  %c = and i1 %a, %b
  ret i1 %c
}

; CHECK-LABEL: define i1 @negative5
; CHECK: %a = icmp
; CHECK: %b = icmp
; CHECK: %c = or i1 %a, %b
; CHECK: ret i1 %c
define i1 @negative5(i32 %x, i32 %n) {
  %nn = and i32 %n, 2147483647
  %a = icmp slt i32 %x, %nn
  %b = icmp sge i32 %x, 0
  %c = or i1 %a, %b            ; should be: and
  ret i1 %c
}

