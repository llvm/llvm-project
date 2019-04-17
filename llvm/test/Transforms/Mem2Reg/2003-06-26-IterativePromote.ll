; Promoting some values allows promotion of other values.
; RUN: opt < %s -mem2reg -S | not grep alloca

define i32 @test2() {
	%result = alloca i32		; <i32*> [#uses=2]
	%a = alloca i32		; <i32*> [#uses=2]
	%p = alloca i32*		; <i32**> [#uses=2]
	store i32 0, i32* %a
	store i32* %a, i32** %p
	%tmp.0 = load i32*, i32** %p		; <i32*> [#uses=1]
	%tmp.1 = load i32, i32* %tmp.0		; <i32> [#uses=1]
	store i32 %tmp.1, i32* %result
	%tmp.2 = load i32, i32* %result		; <i32> [#uses=1]
	ret i32 %tmp.2
}

