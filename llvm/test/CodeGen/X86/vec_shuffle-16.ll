; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse,-sse2 | grep shufps | count 4
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse,-sse2 -mtriple=i386-apple-darwin | grep mov | count 2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pshufd | count 4
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep shufps
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -mtriple=i386-apple-darwin | not grep mov

define <4 x float> @t1(<4 x float> %a, <4 x float> %b) nounwind  {
        %tmp1 = shufflevector <4 x float> %b, <4 x float> undef, <4 x i32> zeroinitializer
        ret <4 x float> %tmp1
}

define <4 x float> @t2(<4 x float> %A, <4 x float> %B) nounwind {
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >
	ret <4 x float> %tmp
}

define <4 x float> @t3(<4 x float> %A, <4 x float> %B) nounwind {
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 4, i32 4, i32 4, i32 4 >
	ret <4 x float> %tmp
}

define <4 x float> @t4(<4 x float> %A, <4 x float> %B) nounwind {
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 1, i32 3, i32 2, i32 0 >
	ret <4 x float> %tmp
}
