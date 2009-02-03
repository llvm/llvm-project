; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movq

define <4 x i32> @t(i32 %x, i32 %y) nounwind  {
	%tmp1 = insertelement <4 x i32> zeroinitializer, i32 %x, i32 0
	%tmp2 = insertelement <4 x i32> %tmp1, i32 %y, i32 1
	ret <4 x i32> %tmp2
}
