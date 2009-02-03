; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep xor | count 2

define <4 x i32> @t1() {
	%tmp = xor <4 x i32> undef, undef
	ret <4 x i32> %tmp
}

define i32 @t2() {
	%tmp = xor i32 undef, undef
	ret i32 %tmp
}
