; RUN: opt < %s -instcombine -S | FileCheck %s
; PR3103

define i8 @test1(i8 %x, i8 %y) {
; CHECK: @test1
  %A = udiv i8 %x, %y
; CHECK-NEXT: urem
  %B = mul i8 %A, %y
  %C = sub i8 %x, %B
  ret i8 %C
; CHECK-NEXT: ret
}

define i8 @test2(i8 %x, i8 %y) {
; CHECK: @test2
  %A = sdiv i8 %x, %y
; CHECK-NEXT: srem
  %B = mul i8 %A, %y
  %C = sub i8 %x, %B
  ret i8 %C
; CHECK-NEXT: ret
}

define i8 @test3(i8 %x, i8 %y) {
; CHECK: @test3
  %A = udiv i8 %x, %y
; CHECK-NEXT: urem
  %B = mul i8 %A, %y
  %C = sub i8 %B, %x
; CHECK-NEXT: sub
  ret i8 %C
; CHECK-NEXT: ret
}

define i8 @test4(i8 %x) {
; CHECK: @test4
  %A = udiv i8 %x, 3
; CHECK-NEXT: urem
  %B = mul i8 %A, -3
; CHECK-NEXT: sub
  %C = sub i8 %x, %B
; CHECK-NEXT: add
  ret i8 %C
; CHECK-NEXT: ret
}

define i32 @test5(i32 %x, i32 %y) {
; CHECK: @test5
; (((X / Y) * Y) / Y) -> X / Y
  %div = sdiv i32 %x, %y
; CHECK-NEXT: sdiv
  %mul = mul i32 %div, %y
  %r = sdiv i32 %mul, %y
  ret i32 %r
; CHECK-NEXT: ret
}

define i32 @test6(i32 %x, i32 %y) {
; CHECK: @test6
; (((X / Y) * Y) / Y) -> X / Y
  %div = udiv i32 %x, %y
; CHECK-NEXT: udiv
  %mul = mul i32 %div, %y
  %r = udiv i32 %mul, %y
  ret i32 %r
; CHECK-NEXT: ret
}
