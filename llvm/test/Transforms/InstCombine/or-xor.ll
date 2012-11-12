; RUN: opt -S -instcombine < %s | FileCheck %s

define i32 @test1(i32 %x, i32 %y) nounwind {
  %or = or i32 %x, %y
  %not = xor i32 %or, -1
  %z = or i32 %x, %not
  ret i32 %z
; CHECK: @test1
; CHECK-NEXT: %y.not = xor i32 %y, -1
; CHECK-NEXT: %z = or i32 %y.not, %x
; CHECK-NEXT: ret i32 %z
}

define i32 @test2(i32 %x, i32 %y) nounwind {
  %or = or i32 %x, %y
  %not = xor i32 %or, -1
  %z = or i32 %y, %not
  ret i32 %z
; CHECK: @test2
; CHECK-NEXT: %x.not = xor i32 %x, -1
; CHECK-NEXT: %z = or i32 %x.not, %y
; CHECK-NEXT: ret i32 %z
}

define i32 @test3(i32 %x, i32 %y) nounwind {
  %xor = xor i32 %x, %y
  %not = xor i32 %xor, -1
  %z = or i32 %x, %not
  ret i32 %z
; CHECK: @test3
; CHECK-NEXT: %y.not = xor i32 %y, -1
; CHECK-NEXT: %z = or i32 %y.not, %x
; CHECK-NEXT: ret i32 %z
}

define i32 @test4(i32 %x, i32 %y) nounwind {
  %xor = xor i32 %x, %y
  %not = xor i32 %xor, -1
  %z = or i32 %y, %not
  ret i32 %z
; CHECK: @test4
; CHECK-NEXT: %x.not = xor i32 %x, -1
; CHECK-NEXT: %z = or i32 %x.not, %y
; CHECK-NEXT: ret i32 %z
}

define i32 @test5(i32 %x, i32 %y) nounwind {
  %and = and i32 %x, %y
  %not = xor i32 %and, -1
  %z = or i32 %x, %not
  ret i32 %z
; CHECK: @test5
; CHECK-NEXT: ret i32 -1
}

define i32 @test6(i32 %x, i32 %y) nounwind {
  %and = and i32 %x, %y
  %not = xor i32 %and, -1
  %z = or i32 %y, %not
  ret i32 %z
; CHECK: @test6
; CHECK-NEXT: ret i32 -1
}

define i32 @test7(i32 %x, i32 %y) nounwind {
  %xor = xor i32 %x, %y
  %z = or i32 %y, %xor
  ret i32 %z
; CHECK: @test7
; CHECK-NEXT: %z = or i32 %x, %y
; CHECK-NEXT: ret i32 %z
}

define i32 @test8(i32 %x, i32 %y) nounwind {
  %not = xor i32 %y, -1
  %xor = xor i32 %x, %not
  %z = or i32 %y, %xor
  ret i32 %z
; CHECK: @test8
; CHECK-NEXT: %x.not = xor i32 %x, -1
; CHECK-NEXT: %z = or i32 %x.not, %y
; CHECK-NEXT: ret i32 %z
}

define i32 @test9(i32 %x, i32 %y) nounwind {
  %not = xor i32 %x, -1
  %xor = xor i32 %not, %y
  %z = or i32 %x, %xor
  ret i32 %z
; CHECK: @test9
; CHECK-NEXT: %y.not = xor i32 %y, -1
; CHECK-NEXT: %z = or i32 %y.not, %x
; CHECK-NEXT: ret i32 %z
}
