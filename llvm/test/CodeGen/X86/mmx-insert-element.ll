; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | not grep movq
; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | grep psllq

define <2 x i32> @qux(i32 %A) nounwind {
	%tmp3 = insertelement <2 x i32> < i32 0, i32 undef >, i32 %A, i32 1		; <<2 x i32>> [#uses=1]
	ret <2 x i32> %tmp3
}
