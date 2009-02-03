; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5

define <4 x i32> @test_arg(<4 x i32> %A, <4 x i32> %B) {
	%C = add <4 x i32> %A, %B		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %C
}

define <4 x i32> @foo() {
	%X = call <4 x i32> @test_arg( <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %X
}
