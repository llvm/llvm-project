; RUN: llvm-as < %s | opt -sccp | llvm-dis | not grep %X

@G = external global [40 x i32]		; <[40 x i32]*> [#uses=1]

define i32* @test() {
	%X = getelementptr [40 x i32]* @G, i64 0, i64 0		; <i32*> [#uses=1]
	ret i32* %X
}

