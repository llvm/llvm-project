; In this test, a local alloca cannot alias an incoming argument.

; RUN: llvm-as < %s | opt -gvn -instcombine | llvm-dis | not grep sub

define i32 @test(i32* %P) {
	%X = alloca i32
	%V1 = load i32* %P
	store i32 0, i32* %X
	%V2 = load i32* %P
	%Diff = sub i32 %V1, %V2
	ret i32 %Diff
}
