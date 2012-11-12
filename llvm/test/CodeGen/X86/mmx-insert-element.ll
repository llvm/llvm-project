; RUN: llc < %s -march=x86 -mattr=+mmx,+sse2 | grep movq
; RUN: llc < %s -march=x86 -mattr=+mmx,+sse2 | grep pshufd
; This is not an MMX operation; promoted to XMM.

define x86_mmx @qux(i32 %A) nounwind {
	%tmp3 = insertelement <2 x i32> < i32 0, i32 undef >, i32 %A, i32 1		; <<2 x i32>> [#uses=1]
        %tmp4 = bitcast <2 x i32> %tmp3 to x86_mmx
	ret x86_mmx %tmp4
}
