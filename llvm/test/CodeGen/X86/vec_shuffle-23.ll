; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2                | not grep punpck
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2                |     grep pshufd

define i32 @t() nounwind {
entry:
	%a = alloca <4 x i32>		; <<4 x i32>*> [#uses=2]
	%b = alloca <4 x i32>		; <<4 x i32>*> [#uses=5]
	volatile store <4 x i32> < i32 0, i32 1, i32 2, i32 3 >, <4 x i32>* %a
	%tmp = load <4 x i32>* %a		; <<4 x i32>> [#uses=1]
	store <4 x i32> %tmp, <4 x i32>* %b
	%tmp1 = load <4 x i32>* %b		; <<4 x i32>> [#uses=1]
	%tmp2 = load <4 x i32>* %b		; <<4 x i32>> [#uses=1]
	%punpckldq = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x i32>> [#uses=1]
	store <4 x i32> %punpckldq, <4 x i32>* %b
	%tmp3 = load <4 x i32>* %b		; <<4 x i32>> [#uses=1]
	%result = extractelement <4 x i32> %tmp3, i32 0		; <i32> [#uses=1]
	ret i32 %result
}
