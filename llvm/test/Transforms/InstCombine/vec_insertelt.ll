; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret <4 x i32> %A}

; PR1286
define <4 x i32> @test1(<4 x i32> %A) {
	%B = insertelement <4 x i32> %A, i32 undef, i32 1
	ret <4 x i32> %B
}
