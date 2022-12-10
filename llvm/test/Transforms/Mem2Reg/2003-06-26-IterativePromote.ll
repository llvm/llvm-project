; Promoting some values allows promotion of other values.
; RUN: opt < %s -passes=mem2reg -S | not grep alloca

define i32 @test2() {
	%result = alloca i32		; <ptr> [#uses=2]
	%a = alloca i32		; <ptr> [#uses=2]
	%p = alloca ptr		; <ptr> [#uses=2]
	store i32 0, ptr %a
	store ptr %a, ptr %p
	%tmp.0 = load ptr, ptr %p		; <ptr> [#uses=1]
	%tmp.1 = load i32, ptr %tmp.0		; <i32> [#uses=1]
	store i32 %tmp.1, ptr %result
	%tmp.2 = load i32, ptr %result		; <i32> [#uses=1]
	ret i32 %tmp.2
}

