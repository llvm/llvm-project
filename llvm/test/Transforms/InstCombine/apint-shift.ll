; This test makes sure that shift instructions are properly eliminated
; even with arbitrary precision integers.
; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: @test1
; CHECK-NOT: sh
define i47 @test1(i47 %A) {
	%B = shl i47 %A, 0		; <i47> [#uses=1]
	ret i47 %B
}

; CHECK: @test2
; CHECK-NOT: sh
define i41 @test2(i7 %X) {
	%A = zext i7 %X to i41		; <i41> [#uses=1]
	%B = shl i41 0, %A		; <i41> [#uses=1]
	ret i41 %B
}

; CHECK: @test3
; CHECK-NOT: sh
define i41 @test3(i41 %A) {
	%B = ashr i41 %A, 0		; <i41> [#uses=1]
	ret i41 %B
}

; CHECK: @test4
; CHECK-NOT: sh
define i39 @test4(i7 %X) {
	%A = zext i7 %X to i39		; <i39> [#uses=1]
	%B = ashr i39 0, %A		; <i39> [#uses=1]
	ret i39 %B
}

; CHECK: @test5
; CHECK-NOT: sh
define i55 @test5(i55 %A) {
	%B = lshr i55 %A, 55		; <i55> [#uses=1]
	ret i55 %B
}

; CHECK: @test5a
; CHECK-NOT: sh
define i32 @test5a(i32 %A) {
	%B = shl i32 %A, 32		; <i32> [#uses=1]
	ret i32 %B
}

; CHECK: @test6
; CHECK: mul i55 %A, 6
define i55 @test6(i55 %A) {
	%B = shl i55 %A, 1		; <i55> [#uses=1]
	%C = mul i55 %B, 3		; <i55> [#uses=1]
	ret i55 %C
}

; CHECK: @test6a
; CHECK: mul i55 %A, 6
define i55 @test6a(i55 %A) {
	%B = mul i55 %A, 3		; <i55> [#uses=1]
	%C = shl i55 %B, 1		; <i55> [#uses=1]
	ret i55 %C
}

; CHECK: @test7
; CHECK-NOT: sh
define i29 @test7(i8 %X) {
	%A = zext i8 %X to i29		; <i29> [#uses=1]
	%B = ashr i29 -1, %A		; <i29> [#uses=1]
	ret i29 %B
}

; CHECK: @test8
; CHECK-NOT: sh
define i7 @test8(i7 %A) {
	%B = shl i7 %A, 4		; <i7> [#uses=1]
	%C = shl i7 %B, 3		; <i7> [#uses=1]
	ret i7 %C
}

; CHECK: @test9
; CHECK-NOT: sh
define i17 @test9(i17 %A) {
	%B = shl i17 %A, 16		; <i17> [#uses=1]
	%C = lshr i17 %B, 16		; <i17> [#uses=1]
	ret i17 %C
}

; CHECK: @test10
; CHECK-NOT: sh
define i19 @test10(i19 %A) {
	%B = lshr i19 %A, 18		; <i19> [#uses=1]
	%C = shl i19 %B, 18		; <i19> [#uses=1]
	ret i19 %C
}

; CHECK: @test11
; Don't hide the shl from scalar evolution. DAGCombine will get it.
; CHECK: shl
define i23 @test11(i23 %A) {
	%a = mul i23 %A, 3		; <i23> [#uses=1]
	%B = lshr i23 %a, 11		; <i23> [#uses=1]
	%C = shl i23 %B, 12		; <i23> [#uses=1]
	ret i23 %C
}

; CHECK: @test12
; CHECK-NOT: sh
define i47 @test12(i47 %A) {
	%B = ashr i47 %A, 8		; <i47> [#uses=1]
	%C = shl i47 %B, 8		; <i47> [#uses=1]
	ret i47 %C
}

; CHECK: @test13
; Don't hide the shl from scalar evolution. DAGCombine will get it.
; CHECK: shl
define i18 @test13(i18 %A) {
	%a = mul i18 %A, 3		; <i18> [#uses=1]
	%B = ashr i18 %a, 8		; <i18> [#uses=1]
	%C = shl i18 %B, 9		; <i18> [#uses=1]
	ret i18 %C
}

; CHECK: @test14
; CHECK-NOT: sh
define i35 @test14(i35 %A) {
	%B = lshr i35 %A, 4		; <i35> [#uses=1]
	%C = or i35 %B, 1234		; <i35> [#uses=1]
	%D = shl i35 %C, 4		; <i35> [#uses=1]
	ret i35 %D
}

; CHECK: @test14a
; CHECK-NOT: sh
define i79 @test14a(i79 %A) {
	%B = shl i79 %A, 4		; <i79> [#uses=1]
	%C = and i79 %B, 1234		; <i79> [#uses=1]
	%D = lshr i79 %C, 4		; <i79> [#uses=1]
	ret i79 %D
}

; CHECK: @test15
; CHECK-NOT: sh
define i45 @test15(i1 %C) {
	%A = select i1 %C, i45 3, i45 1	; <i45> [#uses=1]
	%V = shl i45 %A, 2		; <i45> [#uses=1]
	ret i45 %V
}

; CHECK: @test15a
; CHECK-NOT: sh
define i53 @test15a(i1 %X) {
	%A = select i1 %X, i8 3, i8 1	; <i8> [#uses=1]
	%B = zext i8 %A to i53		; <i53> [#uses=1]
	%V = shl i53 64, %B		; <i53> [#uses=1]
	ret i53 %V
}

; CHECK: @test16
; CHECK-NOT: sh
define i1 @test16(i84 %X) {
	%tmp.3 = ashr i84 %X, 4		; <i84> [#uses=1]
	%tmp.6 = and i84 %tmp.3, 1	; <i84> [#uses=1]
	%tmp.7 = icmp ne i84 %tmp.6, 0	; <i1> [#uses=1]
	ret i1 %tmp.7
}

; CHECK: @test17
; CHECK-NOT: sh
define i1 @test17(i106 %A) {
	%B = lshr i106 %A, 3		; <i106> [#uses=1]
	%C = icmp eq i106 %B, 1234	; <i1> [#uses=1]
	ret i1 %C
}

; CHECK: @test18
; CHECK-NOT: sh
define i1 @test18(i11 %A) {
	%B = lshr i11 %A, 10		; <i11> [#uses=1]
	%C = icmp eq i11 %B, 123	; <i1> [#uses=1]
	ret i1 %C
}

; CHECK: @test19
; CHECK-NOT: sh
define i1 @test19(i37 %A) {
	%B = ashr i37 %A, 2		; <i37> [#uses=1]
	%C = icmp eq i37 %B, 0		; <i1> [#uses=1]
	ret i1 %C
}

; CHECK: @test19a
; CHECK-NOT: sh
define i1 @test19a(i39 %A) {
	%B = ashr i39 %A, 2		; <i39> [#uses=1]
	%C = icmp eq i39 %B, -1		; <i1> [#uses=1]
	ret i1 %C
}

; CHECK: @test20
; CHECK-NOT: sh
define i1 @test20(i13 %A) {
	%B = ashr i13 %A, 12		; <i13> [#uses=1]
	%C = icmp eq i13 %B, 123	; <i1> [#uses=1]
	ret i1 %C
}

; CHECK: @test21
; CHECK-NOT: sh
define i1 @test21(i12 %A) {
	%B = shl i12 %A, 6		; <i12> [#uses=1]
	%C = icmp eq i12 %B, -128		; <i1> [#uses=1]
	ret i1 %C
}

; CHECK: @test22
; CHECK-NOT: sh
define i1 @test22(i14 %A) {
	%B = shl i14 %A, 7		; <i14> [#uses=1]
	%C = icmp eq i14 %B, 0		; <i1> [#uses=1]
	ret i1 %C
}

; CHECK: @test23
; CHECK-NOT: sh
define i11 @test23(i44 %A) {
	%B = shl i44 %A, 33		; <i44> [#uses=1]
	%C = ashr i44 %B, 33		; <i44> [#uses=1]
	%D = trunc i44 %C to i11	; <i8> [#uses=1]
	ret i11 %D
}

; CHECK: @test25
; CHECK-NOT: sh
define i37 @test25(i37 %tmp.2, i37 %AA) {
	%x = lshr i37 %AA, 17		; <i37> [#uses=1]
	%tmp.3 = lshr i37 %tmp.2, 17		; <i37> [#uses=1]
	%tmp.5 = add i37 %tmp.3, %x		; <i37> [#uses=1]
	%tmp.6 = shl i37 %tmp.5, 17		; <i37> [#uses=1]
	ret i37 %tmp.6
}

; CHECK: @test26
; CHECK-NOT: sh
define i40 @test26(i40 %A) {
	%B = lshr i40 %A, 1		; <i40> [#uses=1]
	%C = bitcast i40 %B to i40		; <i40> [#uses=1]
	%D = shl i40 %C, 1		; <i40> [#uses=1]
	ret i40 %D
}
