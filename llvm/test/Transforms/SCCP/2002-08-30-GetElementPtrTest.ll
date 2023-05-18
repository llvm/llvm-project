; RUN: opt < %s -passes=sccp -S | not grep %X

@G = external global [40 x i32]		; <ptr> [#uses=1]

define ptr @test() {
	%X = getelementptr [40 x i32], ptr @G, i64 0, i64 0		; <ptr> [#uses=1]
	ret ptr %X
}

