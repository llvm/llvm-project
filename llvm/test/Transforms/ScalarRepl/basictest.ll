; RUN: llvm-as < %s | opt -scalarrepl -mem2reg | llvm-dis | not grep alloca

define i32 @test() {
	%X = alloca { i32, float }		; <{ i32, float }*> [#uses=1]
	%Y = getelementptr { i32, float }* %X, i64 0, i32 0		; <i32*> [#uses=2]
	store i32 0, i32* %Y
	%Z = load i32* %Y		; <i32> [#uses=1]
	ret i32 %Z
}

