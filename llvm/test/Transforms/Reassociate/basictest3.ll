; RUN: llvm-as < %s | opt -reassociate -gvn | llvm-dis | grep add | count 6
; Each of these functions should turn into two adds each.

@e = external global i32		; <i32*> [#uses=3]
@a = external global i32		; <i32*> [#uses=3]
@b = external global i32		; <i32*> [#uses=3]
@c = external global i32		; <i32*> [#uses=3]
@f = external global i32		; <i32*> [#uses=3]

define void @test1() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %A, %B		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = (a+b)+c;
        store i32 %t2, i32* @e
        ; f = (a+c)+b
	store i32 %t4, i32* @f
	ret void
}

define void @test2() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %A, %B		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = c+(a+b)
        store i32 %t2, i32* @e
        ; f = (c+a)+b
	store i32 %t4, i32* @f
	ret void
}

define void @test3() {
	%A = load i32* @a		; <i32> [#uses=2]
	%B = load i32* @b		; <i32> [#uses=2]
	%C = load i32* @c		; <i32> [#uses=2]
	%t1 = add i32 %B, %A		; <i32> [#uses=1]
	%t2 = add i32 %t1, %C		; <i32> [#uses=1]
	%t3 = add i32 %C, %A		; <i32> [#uses=1]
	%t4 = add i32 %t3, %B		; <i32> [#uses=1]
	; e = c+(b+a)
        store i32 %t2, i32* @e
        ; f = (c+a)+b
	store i32 %t4, i32* @f
	ret void
}

