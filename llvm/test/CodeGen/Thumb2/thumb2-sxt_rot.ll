; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2,+t2xtpk %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m3 %s -o - | FileCheck %s --check-prefix=CHECK-M3

define i32 @test0(i8 %A) {
; CHECK-LABEL: test0:
; CHECK: sxtb r0, r0
; CHECK-M3: sxtb r0, r0
        %B = sext i8 %A to i32
	ret i32 %B
}

define signext i8 @test1(i32 %A)  {
; CHECK-LABEL: test1:
; CHECK: sbfx r0, r0, #8, #8
; CHECK-M3: sbfx r0, r0, #8, #8
	%B = lshr i32 %A, 8
	%C = shl i32 %A, 24
	%D = or i32 %B, %C
	%E = trunc i32 %D to i8
	ret i8 %E
}

define signext i32 @test2(i32 %A, i32 %X)  {
; CHECK-LABEL: test2:
; CHECK: sxtab  r0, r1, r0, ror #8
; CHECK-M3-NOT: sxtab
	%B = lshr i32 %A, 8
	%C = shl i32 %A, 24
	%D = or i32 %B, %C
	%E = trunc i32 %D to i8
        %F = sext i8 %E to i32
        %G = add i32 %F, %X
	ret i32 %G
}

define i32 @test3(i32 %A, i32 %X) {
; CHECK-LABEL: test3:
; CHECK: sxtah r0, r0, r1, ror #8
; CHECK-M3-NOT: sxtah
  %X.hi = lshr i32 %X, 8
  %X.trunc = trunc i32 %X.hi to i16
  %addend = sext i16 %X.trunc to i32
  %sum = add i32 %A, %addend
  ret i32 %sum
}

define signext i32 @test4(i32 %A, i32 %X)  {
; CHECK-LABEL: test4:
; CHECK: sxtab  r0, r1, r0, ror #16
; CHECK-M3-NOT: sxtab
	%B = lshr i32 %A, 16
	%C = shl i32 %A, 16
	%D = or i32 %B, %C
	%E = trunc i32 %D to i8
        %F = sext i8 %E to i32
        %G = add i32 %F, %X
	ret i32 %G
}

define signext i32 @test5(i32 %A, i32 %X)  {
; CHECK-LABEL: test5:
; CHECK: sxtah  r0, r1, r0, ror #24
; CHECK-M3-NOT: sxtah
	%B = lshr i32 %A, 24
	%C = shl i32 %A, 8
	%D = or i32 %B, %C
	%E = trunc i32 %D to i16
        %F = sext i16 %E to i32
        %G = add i32 %F, %X
	ret i32 %G
}
