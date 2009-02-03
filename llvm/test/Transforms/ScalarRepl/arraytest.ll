; RUN: llvm-as < %s | opt -scalarrepl -mem2reg | llvm-dis | not grep alloca

define i32 @test() {
	%X = alloca [4 x i32]		; <[4 x i32]*> [#uses=1]
	%Y = getelementptr [4 x i32]* %X, i64 0, i64 0		; <i32*> [#uses=2]
	store i32 0, i32* %Y
	%Z = load i32* %Y		; <i32> [#uses=1]
	ret i32 %Z
}

