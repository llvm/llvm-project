; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s --check-prefix=A8
; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m3 %s -o - | FileCheck %s --check-prefix=M3
; rdar://11318438

define zeroext i8 @test1(i32 %A.u)  {
; CHECK-LABEL: test1:
; A8: uxtb r0, r0
    %B.u = trunc i32 %A.u to i8
    ret i8 %B.u
}

define zeroext i32 @test2(i32 %A.u, i32 %B.u)  {
; CHECK-LABEL: test2:
; A8: uxtab  r0, r0, r1

; M3: uxtb  r1, r1
; M3-NOT: uxtab
; M3: add   r0, r1
    %C.u = trunc i32 %B.u to i8
    %D.u = zext i8 %C.u to i32
    %E.u = add i32 %A.u, %D.u
    ret i32 %E.u
}

define zeroext i32 @test3(i32 %A.u)  {
; CHECK-LABEL: test3:
; A8: ubfx  r0, r0, #8, #16
; M3: ubfx r0, r0, #8, #16
    %B.u = lshr i32 %A.u, 8
    %C.u = shl i32 %A.u, 24
    %D.u = or i32 %B.u, %C.u
    %E.u = trunc i32 %D.u to i16
    %F.u = zext i16 %E.u to i32
    ret i32 %F.u
}

define i32 @test4(i32 %A, i32 %X) {
; CHECK-LABEL: test4:
; A8: uxtab r0, r0, r1, ror #16
; M3-NOT: uxtab
  %X.hi = lshr i32 %X, 16
  %X.trunc = trunc i32 %X.hi to i8
  %addend = zext i8 %X.trunc to i32
  %sum = add i32 %A, %addend
  ret i32 %sum
}

define i32 @test5(i32 %A, i32 %X) {
; CHECK-LABEL: test5:
; A8: uxtah r0, r0, r1, ror #8
; M3-NOT: uxtah
  %X.hi = lshr i32 %X, 8
  %X.trunc = trunc i32 %X.hi to i16
  %addend = zext i16 %X.trunc to i32
  %sum = add i32 %A, %addend
  ret i32 %sum
}

define i32 @test6(i32 %A, i32 %X) {
; CHECK-LABEL: test6:
; A8: uxtab r0, r0, r1, ror #8
; M3-NOT: uxtab
  %X.hi = lshr i32 %X, 8
  %X.trunc = trunc i32 %X.hi to i8
  %addend = zext i8 %X.trunc to i32
  %sum = add i32 %A, %addend
  ret i32 %sum
}

define i32 @test7(i32 %A, i32 %X) {
; CHECK-LABEL: test7:
; A8: uxtah r0, r0, r1, ror #24
; M3-NOT: uxtah
  %lshr = lshr i32 %X, 24
  %shl = shl i32 %X, 8
  %or = or i32 %lshr, %shl
  %trunc = trunc i32 %or to i16
  %zext = zext i16 %trunc to i32
  %add = add i32 %A, %zext
  ret i32 %add
}

define i32 @test8(i32 %A, i32 %X) {
; CHECK-LABEL: test8:
; A8: uxtah r0, r0, r1, ror #24
; M3-NOT: uxtah
  %lshr = lshr i32 %X, 24
  %shl = shl i32 %X, 8
  %or = or i32 %lshr, %shl
  %and = and i32 %or, 65535
  %add = add i32 %A, %and
  ret i32 %add
}
