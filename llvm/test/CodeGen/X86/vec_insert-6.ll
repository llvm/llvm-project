; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=penryn | grep pslldq
; RUN: llc < %s -march=x86 -mattr=+sse2 -mcpu=penryn -mtriple=i686-apple-darwin9 -o /dev/null -stats -info-output-file - | grep asm-printer | grep 6

define <4 x float> @t3(<4 x float>* %P) nounwind  {
	%tmp1 = load <4 x float>* %P
	%tmp2 = shufflevector <4 x float> zeroinitializer, <4 x float> %tmp1, <4 x i32> < i32 4, i32 4, i32 4, i32 0 >
	ret <4 x float> %tmp2
}
