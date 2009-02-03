; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2       | not grep shuf
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2,-sse3 | grep movlhps | count 2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse3       | grep movddup | count 1

define <4 x float> @t1(<4 x float> %a) nounwind  {
entry:
        %tmp1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> < i32 0, i32 1, i32 0, i32 1 >       ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp1
}

define <4 x i32> @t2(<4 x i32>* %a) nounwind {
entry:
        %tmp1 = load <4 x i32>* %a;
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> < i32 0, i32 1, i32 0, i32 1 >		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp2
}
