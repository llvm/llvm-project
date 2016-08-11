; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o - | FileCheck %s --check-prefix=CHECK-V6
; RUN: llc -mtriple=arm-eabi -mattr=+v7 %s -o - | FileCheck %s --check-prefix=CHECK-V7

define zeroext i8 @test1(i32 %A.u) {
  ; CHECK-LABEL: test1
  ; CHECK-V6: uxtb
  ; CHECK-V7: uxtb
    %B.u = trunc i32 %A.u to i8
    ret i8 %B.u
}

define zeroext i32 @test2(i32 %A.u, i32 %B.u) {
  ; CHECK-LABEL: test2
  ; CHECK-V6: uxtab r0, r0, r1
  ; CHECK-V7: uxtab r0, r0, r1
    %C.u = trunc i32 %B.u to i8
    %D.u = zext i8 %C.u to i32
    %E.u = add i32 %A.u, %D.u
    ret i32 %E.u
}

define zeroext i32 @test3(i32 %A.u) {
  ; CHECK-LABEL: test3
  ; CHECK-V6-NOT: ubfx
  ; CHECK-V7: ubfx r0, r0, #8, #16
    %B.u = lshr i32 %A.u, 8
    %C.u = shl i32 %A.u, 24
    %D.u = or i32 %B.u, %C.u
    %E.u = trunc i32 %D.u to i16
    %F.u = zext i16 %E.u to i32
    ret i32 %F.u
}

define zeroext i32 @test4(i32 %A.u) {
  ; CHECK-LABEL: test4
  ; CHECK-V6-NOT: ubfx
  ; CHECK-V7: ubfx r0, r0, #8, #8
    %B.u = lshr i32 %A.u, 8
    %C.u = shl i32 %A.u, 24
    %D.u = or i32 %B.u, %C.u
    %E.u = trunc i32 %D.u to i8
    %F.u = zext i8 %E.u to i32
    ret i32 %F.u
}

define zeroext i16 @test5(i32 %A.u) {
  ; CHECK-LABEL: test5
  ; CHECK-V6: uxth
  ; CHECK-V7: uxth
    %B.u = trunc i32 %A.u to i16
    ret i16 %B.u
}

define zeroext i32 @test6(i32 %A.u, i32 %B.u) {
  ; CHECK-LABEL: test6
  ; CHECK-V6: uxtah r0, r0, r1
  ; CHECK-V7: uxtah r0, r0, r1
    %C.u = trunc i32 %B.u to i16
    %D.u = zext i16 %C.u to i32
    %E.u = add i32 %A.u, %D.u
    ret i32 %E.u
}

define zeroext i32 @test7(i32 %A, i32 %X) {
; CHECK-LABEL: test7
; CHECK-V6: uxtab r0, r1, r0, ror #8
; CHECK-V7: uxtab r0, r1, r0, ror #8
  %B = lshr i32 %A, 8
  %C = shl i32 %A, 24
  %D = or i32 %B, %C
  %E = trunc i32 %D to i8
  %F = zext i8 %E to i32
  %G = add i32 %F, %X
  ret i32 %G
}

define zeroext i32 @test8(i32 %A, i32 %X) {
; CHECK-LABEL: test8
; CHECK-V6: uxtab r0, r1, r0, ror #16
; CHECK-V7: uxtab r0, r1, r0, ror #16
  %B = lshr i32 %A, 16
  %C = shl i32 %A, 16
  %D = or i32 %B, %C
  %E = trunc i32 %D to i8
  %F = zext i8 %E to i32
  %G = add i32 %F, %X
  ret i32 %G
}

define zeroext i32 @test9(i32 %A, i32 %X) {
; CHECK-LABEL: test9
; CHECK-V6: uxtah r0, r1, r0, ror #8
; CHECK-V7: uxtah r0, r1, r0, ror #8
  %B = lshr i32 %A, 8
  %C = shl i32 %A, 24
  %D = or i32 %B, %C
  %E = trunc i32 %D to i16
  %F = zext i16 %E to i32
  %G = add i32 %F, %X
  ret i32 %G
}

define zeroext i32 @test10(i32 %A, i32 %X) {
; CHECK-LABEL: test10
; CHECK-V6: uxtah r0, r1, r0, ror #24
; CHECK-V7: uxtah r0, r1, r0, ror #24
  %B = lshr i32 %A, 24
  %C = shl i32 %A, 8
  %D = or i32 %B, %C
  %E = trunc i32 %D to i16
  %F = zext i16 %E to i32
  %G = add i32 %F, %X
  ret i32 %G
}

define zeroext i32 @test11(i32 %A, i32 %X) {
; CHECK-LABEL: test11
; CHECK-V6: uxtab r0, r1, r0
; CHECK-V7: uxtab r0, r1, r0
  %B = and i32 %A, 255
  %add = add i32 %X, %B
  ret i32 %add
}

define zeroext i32 @test12(i32 %A, i32 %X) {
; CHECK-LABEL: test12
; CHECK-V6: uxtab r0, r1, r0, ror #8
; CHECK-V7: uxtab r0, r1, r0, ror #8
  %B = lshr i32 %A, 8
  %and = and i32 %B, 255
  %add = add i32 %and, %X
  ret i32 %add
}

define zeroext i32 @test13(i32 %A, i32 %X) {
; CHECK-LABEL: test13
; CHECK-V6: uxtab r0, r1, r0, ror #16
; CHECK-V7: uxtab r0, r1, r0, ror #16
  %B = lshr i32 %A, 16
  %and = and i32 %B, 255
  %add = add i32 %and, %X
  ret i32 %add
}

define zeroext i32 @test14(i32 %A, i32 %X) {
; CHECK-LABEL: test14
; CHECK-V6: uxtah r0, r1, r0
; CHECK-V7: uxtah r0, r1, r0
  %B = and i32 %A, 65535
  %add = add i32 %X, %B
  ret i32 %add
}

define zeroext i32 @test15(i32 %A, i32 %X) {
; CHECK-LABEL: test15
; CHECK-V6: uxtah r0, r1, r0, ror #8
; CHECK-V7: uxtah r0, r1, r0, ror #8
  %B = lshr i32 %A, 8
  %and = and i32 %B, 65535
  %add = add i32 %and, %X
  ret i32 %add
}

define zeroext i32 @test16(i32 %A, i32 %X) {
; CHECK-LABEL: test16
; CHECK-V6: uxtah r0, r1, r0, ror #24
; CHECK-V7: uxtah r0, r1, r0, ror #24
  %B = lshr i32 %A, 24
  %C = shl i32 %A, 8
  %D = or i32 %B, %C
  %E = and i32 %D, 65535
  %F = add i32 %E, %X
  ret i32 %F
}
