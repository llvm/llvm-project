; RUN: opt < %s -instcombine -S | FileCheck %s

define i41 @test0(i41 %A, i41 %B, i41 %C) {
	%X = shl i41 %A, %C
	%Y = shl i41 %B, %C
	%Z = and i41 %X, %Y
	ret i41 %Z
; CHECK-LABEL: @test0(
; CHECK-NEXT: and i41 %A, %B
; CHECK-NEXT: shl i41
; CHECK-NEXT: ret
}

define i57 @test1(i57 %A, i57 %B, i57 %C) {
	%X = lshr i57 %A, %C
	%Y = lshr i57 %B, %C
	%Z = or i57 %X, %Y
	ret i57 %Z
; CHECK-LABEL: @test1(
; CHECK-NEXT: or i57 %A, %B
; CHECK-NEXT: lshr i57
; CHECK-NEXT: ret
}

define i49 @test2(i49 %A, i49 %B, i49 %C) {
	%X = ashr i49 %A, %C
	%Y = ashr i49 %B, %C
	%Z = xor i49 %X, %Y
	ret i49 %Z
; CHECK-LABEL: @test2(
; CHECK-NEXT: xor i49 %A, %B
; CHECK-NEXT: ashr i49
; CHECK-NEXT: ret
}
