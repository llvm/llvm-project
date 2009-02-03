; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep -v {sub i32 %Cok, %Bok} | grep -v {sub i32 0, %Aok} | not grep sub

define i32 @test1(i32 %A) {
	%B = sub i32 %A, %A		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @test2(i32 %A) {
	%B = sub i32 %A, 0		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @test3(i32 %A) {
	%B = sub i32 0, %A		; <i32> [#uses=1]
	%C = sub i32 0, %B		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test4(i32 %A, i32 %x) {
	%B = sub i32 0, %A		; <i32> [#uses=1]
	%C = sub i32 %x, %B		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test5(i32 %A, i32 %Bok, i32 %Cok) {
	%D = sub i32 %Bok, %Cok		; <i32> [#uses=1]
	%E = sub i32 %A, %D		; <i32> [#uses=1]
	ret i32 %E
}

define i32 @test6(i32 %A, i32 %B) {
	%C = and i32 %A, %B		; <i32> [#uses=1]
	%D = sub i32 %A, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @test7(i32 %A) {
	%B = sub i32 -1, %A		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @test8(i32 %A) {
	%B = mul i32 9, %A		; <i32> [#uses=1]
	%C = sub i32 %B, %A		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test9(i32 %A) {
	%B = mul i32 3, %A		; <i32> [#uses=1]
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test10(i32 %A, i32 %B) {
	%C = sub i32 0, %A		; <i32> [#uses=1]
	%D = sub i32 0, %B		; <i32> [#uses=1]
	%E = mul i32 %C, %D		; <i32> [#uses=1]
	ret i32 %E
}

define i32 @test10.upgrd.1(i32 %A) {
	%C = sub i32 0, %A		; <i32> [#uses=1]
	%E = mul i32 %C, 7		; <i32> [#uses=1]
	ret i32 %E
}

define i1 @test11(i8 %A, i8 %B) {
	%C = sub i8 %A, %B		; <i8> [#uses=1]
	%cD = icmp ne i8 %C, 0		; <i1> [#uses=1]
	ret i1 %cD
}

define i32 @test12(i32 %A) {
	%B = ashr i32 %A, 31		; <i32> [#uses=1]
	%C = sub i32 0, %B		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test13(i32 %A) {
	%B = lshr i32 %A, 31		; <i32> [#uses=1]
	%C = sub i32 0, %B		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test14(i32 %A) {
	%B = lshr i32 %A, 31		; <i32> [#uses=1]
	%C = bitcast i32 %B to i32		; <i32> [#uses=1]
	%D = sub i32 0, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @test15(i32 %A, i32 %B) {
	%C = sub i32 0, %A		; <i32> [#uses=1]
	%D = srem i32 %B, %C		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @test16(i32 %A) {
	%X = sdiv i32 %A, 1123		; <i32> [#uses=1]
	%Y = sub i32 0, %X		; <i32> [#uses=1]
	ret i32 %Y
}

; Can't fold subtract here because negation it might oveflow.
; PR3142
define i32 @test17(i32 %Aok) {
	%B = sub i32 0, %Aok		; <i32> [#uses=1]
	%C = sdiv i32 %B, 1234		; <i32> [#uses=1]
	ret i32 %C
}

define i64 @test18(i64 %Y) {
	%tmp.4 = shl i64 %Y, 2		; <i64> [#uses=1]
	%tmp.12 = shl i64 %Y, 2		; <i64> [#uses=1]
	%tmp.8 = sub i64 %tmp.4, %tmp.12		; <i64> [#uses=1]
	ret i64 %tmp.8
}

define i32 @test19(i32 %X, i32 %Y) {
	%Z = sub i32 %X, %Y		; <i32> [#uses=1]
	%Q = add i32 %Z, %Y		; <i32> [#uses=1]
	ret i32 %Q
}

define i1 @test20(i32 %g, i32 %h) {
	%tmp.2 = sub i32 %g, %h		; <i32> [#uses=1]
	%tmp.4 = icmp ne i32 %tmp.2, %g		; <i1> [#uses=1]
	ret i1 %tmp.4
}

define i1 @test21(i32 %g, i32 %h) {
	%tmp.2 = sub i32 %g, %h		; <i32> [#uses=1]
	%tmp.4 = icmp ne i32 %tmp.2, %g		; <i1> [#uses=1]
	ret i1 %tmp.4
}

; PR2298
define i8 @test22(i32 %a, i32 %b) zeroext nounwind  {
	%tmp2 = sub i32 0, %a		; <i32> [#uses=1]
	%tmp4 = sub i32 0, %b		; <i32> [#uses=1]
	%tmp5 = icmp eq i32 %tmp2, %tmp4		; <i1> [#uses=1]
	%retval89 = zext i1 %tmp5 to i8		; <i8> [#uses=1]
	ret i8 %retval89
}

