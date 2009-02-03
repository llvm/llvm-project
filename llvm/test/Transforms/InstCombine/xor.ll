; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep {xor }
; END.
@G1 = global i32 0		; <i32*> [#uses=1]
@G2 = global i32 0		; <i32*> [#uses=1]

define i1 @test0(i1 %A) {
	%B = xor i1 %A, false		; <i1> [#uses=1]
	ret i1 %B
}

define i32 @test1(i32 %A) {
	%B = xor i32 %A, 0		; <i32> [#uses=1]
	ret i32 %B
}

define i1 @test2(i1 %A) {
	%B = xor i1 %A, %A		; <i1> [#uses=1]
	ret i1 %B
}

define i32 @test3(i32 %A) {
	%B = xor i32 %A, %A		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @test4(i32 %A) {
	%NotA = xor i32 -1, %A		; <i32> [#uses=1]
	%B = xor i32 %A, %NotA		; <i32> [#uses=1]
	ret i32 %B
}

define i32 @test5(i32 %A) {
	%t1 = or i32 %A, 123		; <i32> [#uses=1]
	%r = xor i32 %t1, 123		; <i32> [#uses=1]
	ret i32 %r
}

define i8 @test6(i8 %A) {
	%B = xor i8 %A, 17		; <i8> [#uses=1]
	%C = xor i8 %B, 17		; <i8> [#uses=1]
	ret i8 %C
}

define i32 @test7(i32 %A, i32 %B) {
	%A1 = and i32 %A, 7		; <i32> [#uses=1]
	%B1 = and i32 %B, 128		; <i32> [#uses=1]
	%C1 = xor i32 %A1, %B1		; <i32> [#uses=1]
	ret i32 %C1
}

define i8 @test8(i1 %c) {
	%d = xor i1 %c, true		; <i1> [#uses=1]
	br i1 %d, label %True, label %False

True:		; preds = %0
	ret i8 1

False:		; preds = %0
	ret i8 3
}

define i1 @test9(i8 %A) {
	%B = xor i8 %A, 123		; <i8> [#uses=1]
	%C = icmp eq i8 %B, 34		; <i1> [#uses=1]
	ret i1 %C
}

define i8 @test10(i8 %A) {
	%B = and i8 %A, 3		; <i8> [#uses=1]
	%C = xor i8 %B, 4		; <i8> [#uses=1]
	ret i8 %C
}

define i8 @test11(i8 %A) {
	%B = or i8 %A, 12		; <i8> [#uses=1]
	%C = xor i8 %B, 4		; <i8> [#uses=1]
	ret i8 %C
}

define i1 @test12(i8 %A) {
	%B = xor i8 %A, 4		; <i8> [#uses=1]
	%c = icmp ne i8 %B, 0		; <i1> [#uses=1]
	ret i1 %c
}

define i1 @test13(i8 %A, i8 %B) {
	%C = icmp ult i8 %A, %B		; <i1> [#uses=1]
	%D = icmp ugt i8 %A, %B		; <i1> [#uses=1]
	%E = xor i1 %C, %D		; <i1> [#uses=1]
	ret i1 %E
}

define i1 @test14(i8 %A, i8 %B) {
	%C = icmp eq i8 %A, %B		; <i1> [#uses=1]
	%D = icmp ne i8 %B, %A		; <i1> [#uses=1]
	%E = xor i1 %C, %D		; <i1> [#uses=1]
	ret i1 %E
}

define i32 @test15(i32 %A) {
	%B = add i32 %A, -1		; <i32> [#uses=1]
	%C = xor i32 %B, -1		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test16(i32 %A) {
	%B = add i32 %A, 123		; <i32> [#uses=1]
	%C = xor i32 %B, -1		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test17(i32 %A) {
	%B = sub i32 123, %A		; <i32> [#uses=1]
	%C = xor i32 %B, -1		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test18(i32 %A) {
	%B = xor i32 %A, -1		; <i32> [#uses=1]
	%C = sub i32 123, %B		; <i32> [#uses=1]
	ret i32 %C
}

define i32 @test19(i32 %A, i32 %B) {
	%C = xor i32 %A, %B		; <i32> [#uses=1]
	%D = xor i32 %C, %A		; <i32> [#uses=1]
	ret i32 %D
}

define void @test20(i32 %A, i32 %B) {
	%tmp.2 = xor i32 %B, %A		; <i32> [#uses=2]
	%tmp.5 = xor i32 %tmp.2, %B		; <i32> [#uses=2]
	%tmp.8 = xor i32 %tmp.5, %tmp.2		; <i32> [#uses=1]
	store i32 %tmp.8, i32* @G1
	store i32 %tmp.5, i32* @G2
	ret void
}

define i32 @test21(i1 %C, i32 %A, i32 %B) {
	%C2 = xor i1 %C, true		; <i1> [#uses=1]
	%D = select i1 %C2, i32 %A, i32 %B		; <i32> [#uses=1]
	ret i32 %D
}

define i32 @test22(i1 %X) {
	%Y = xor i1 %X, true		; <i1> [#uses=1]
	%Z = zext i1 %Y to i32		; <i32> [#uses=1]
	%Q = xor i32 %Z, 1		; <i32> [#uses=1]
	ret i32 %Q
}

define i1 @test23(i32 %a, i32 %b) {
	%tmp.2 = xor i32 %b, %a		; <i32> [#uses=1]
	%tmp.4 = icmp eq i32 %tmp.2, %a		; <i1> [#uses=1]
	ret i1 %tmp.4
}

define i1 @test24(i32 %c, i32 %d) {
	%tmp.2 = xor i32 %d, %c		; <i32> [#uses=1]
	%tmp.4 = icmp ne i32 %tmp.2, %c		; <i1> [#uses=1]
	ret i1 %tmp.4
}

define i32 @test25(i32 %g, i32 %h) {
	%h2 = xor i32 %h, -1		; <i32> [#uses=1]
	%tmp2 = and i32 %h2, %g		; <i32> [#uses=1]
	%tmp4 = xor i32 %tmp2, %g		; <i32> [#uses=1]
	ret i32 %tmp4
}

define i32 @test26(i32 %a, i32 %b) {
	%b2 = xor i32 %b, -1		; <i32> [#uses=1]
	%tmp2 = xor i32 %a, %b2		; <i32> [#uses=1]
	%tmp4 = and i32 %tmp2, %a		; <i32> [#uses=1]
	ret i32 %tmp4
}

define i32 @test27(i32 %b, i32 %c, i32 %d) {
	%tmp2 = xor i32 %d, %b		; <i32> [#uses=1]
	%tmp5 = xor i32 %d, %c		; <i32> [#uses=1]
	%tmp = icmp eq i32 %tmp2, %tmp5		; <i1> [#uses=1]
	%tmp6 = zext i1 %tmp to i32		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test28(i32 %indvar) {
	%tmp7 = add i32 %indvar, -2147483647		; <i32> [#uses=1]
	%tmp214 = xor i32 %tmp7, -2147483648		; <i32> [#uses=1]
	ret i32 %tmp214
}
