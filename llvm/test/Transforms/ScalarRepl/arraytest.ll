; RUN: opt < %s -scalarrepl -mem2reg -S | not grep alloca
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define i32 @test() {
	%X = alloca [4 x i32]		; <[4 x i32]*> [#uses=1]
	%Y = getelementptr [4 x i32]* %X, i64 0, i64 0		; <i32*> [#uses=2]
	store i32 0, i32* %Y
	%Z = load i32* %Y		; <i32> [#uses=1]
	ret i32 %Z
}

