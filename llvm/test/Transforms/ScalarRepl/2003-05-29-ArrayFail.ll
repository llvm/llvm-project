; RUN: opt < %s -scalarrepl -instcombine -S | not grep alloca
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

; Test that an array is not incorrectly deconstructed.

define i32 @test() nounwind {
	%X = alloca [4 x i32]		; <[4 x i32]*> [#uses=1]
	%Y = getelementptr [4 x i32]* %X, i64 0, i64 0		; <i32*> [#uses=1]
        ; Must preserve arrayness!
	%Z = getelementptr i32* %Y, i64 1		; <i32*> [#uses=1]
	%A = load i32* %Z		; <i32> [#uses=1]
	ret i32 %A
}
