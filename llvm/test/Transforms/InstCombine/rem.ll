; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep rem
; END.

define i32 @test1(i32 %A) {
	%B = srem i32 %A, 1	; ISA constant 0
	ret i32 %B
}

define i32 @test2(i32 %A) {	; 0 % X = 0, we don't need to preserve traps
	%B = srem i32 0, %A
	ret i32 %B
}

define i32 @test3(i32 %A) {
	%B = urem i32 %A, 8
	ret i32 %B
}

define i1 @test3a(i32 %A) {
	%B = srem i32 %A, -8
	%C = icmp ne i32 %B, 0
	ret i1 %C
}

define i32 @test4(i32 %X, i1 %C) {
	%V = select i1 %C, i32 1, i32 8
	%R = urem i32 %X, %V
	ret i32 %R
}

define i32 @test5(i32 %X, i8 %B) {
	%shift.upgrd.1 = zext i8 %B to i32
	%Amt = shl i32 32, %shift.upgrd.1
	%V = urem i32 %X, %Amt
	ret i32 %V
}

define i32 @test6(i32 %A) {
	%B = srem i32 %A, 0	;; undef
	ret i32 %B
}

define i32 @test7(i32 %A) {
	%B = mul i32 %A, 8
	%C = srem i32 %B, 4
	ret i32 %C
}

define i32 @test8(i32 %A) {
	%B = shl i32 %A, 4
	%C = srem i32 %B, 8
	ret i32 %C
}

define i32 @test9(i32 %A) {
	%B = mul i32 %A, 64
	%C = urem i32 %B, 32
	ret i32 %C
}

define i32 @test10(i8 %c) {
	%tmp.1 = zext i8 %c to i32
	%tmp.2 = mul i32 %tmp.1, 4
	%tmp.3 = sext i32 %tmp.2 to i64
	%tmp.5 = urem i64 %tmp.3, 4
	%tmp.6 = trunc i64 %tmp.5 to i32
	ret i32 %tmp.6
}

define i32 @test11(i32 %i) {
	%tmp.1 = and i32 %i, -2
	%tmp.3 = mul i32 %tmp.1, 2
	%tmp.5 = urem i32 %tmp.3, 4
	ret i32 %tmp.5
}

define i32 @test12(i32 %i) {
	%tmp.1 = and i32 %i, -4
	%tmp.5 = srem i32 %tmp.1, 2
	ret i32 %tmp.5
}
