; RUN: opt -analyze -scalar-evolution %s -S | FileCheck %s

define i16 @test1(i8 %x) {
  %A = zext i8 %x to i12
  %B = sext i12 %A to i16
; CHECK: zext i8 %x to i16
  ret i16 %B
}

define i8 @test2(i8 %x) {
  %A = zext i8 %x to i16
  %B = add i16 %A, 1025
  %C = trunc i16 %B to i8
; CHECK: (1 + %x)
  ret i8 %C
}

define i8 @test3(i8 %x) {
  %A = zext i8 %x to i16
  %B = mul i16 %A, 1027
  %C = trunc i16 %B to i8
; CHECK: (3 * %x)
  ret i8 %C
}

define void @test4(i32 %x, i32 %y) {
entry:
  %Y = and i32 %y, 3
  br label %loop
loop:
  %A = phi i32 [0, %entry], [%I, %loop]
  %rand1 = icmp sgt i32 %A, %Y
  %Z1 = select i1 %rand1, i32 %A, i32 %Y
  %rand2 = icmp ugt i32 %A, %Z1
  %Z2 = select i1 %rand2, i32 %A, i32 %Z1
; CHECK: %Z2 =
; CHECK-NEXT: -->  ([[EXPR:.*]]){{ +}}Exits: 20
  %B = trunc i32 %Z2 to i16
  %C = sext i16 %B to i30
; CHECK: %C =
; CHECK-NEXT: (trunc i32 ([[EXPR]]) to i30)
  %D = sext i16 %B to i32
; CHECK: %D =
; CHECK-NEXT: ([[EXPR]])
  %E = sext i16 %B to i34
; CHECK: %E =
; CHECK-NEXT: (zext i32 ([[EXPR]]) to i34)
  %F = zext i16 %B to i30
; CHECK: %F =
; CHECK-NEXT: (trunc i32 ([[EXPR]]) to i30
  %G = zext i16 %B to i32
; CHECK: %G =
; CHECK-NEXT: ([[EXPR]])
  %H = zext i16 %B to i34
; CHECK: %H =
; CHECK-NEXT: (zext i32 ([[EXPR]]) to i34)
  %I = add i32 %A, 1
  %0 = icmp ne i32 %A, 20
  br i1 %0, label %loop, label %exit
exit:
  ret void
}
