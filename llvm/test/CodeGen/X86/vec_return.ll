; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 > %t
; RUN: grep xorps %t | count 1
; RUN: grep movaps %t | count 1
; RUN: not grep shuf %t

define <2 x double> @test() {
	ret <2 x double> zeroinitializer
}

define <4 x i32> @test2() nounwind  {
	ret <4 x i32> < i32 0, i32 0, i32 1, i32 0 >
}
