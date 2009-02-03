; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx -mtriple=i686-apple-darwin9 -o - | grep punpckldq

define <2 x i32> @mmx_movzl(<2 x i32> %x) nounwind  {
entry:
	%tmp3 = insertelement <2 x i32> %x, i32 32, i32 0		; <<2 x i32>> [#uses=1]
	%tmp8 = insertelement <2 x i32> %tmp3, i32 0, i32 1		; <<2 x i32>> [#uses=1]
	ret <2 x i32> %tmp8
}
