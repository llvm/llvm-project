; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movd | count 1
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+sse2 | grep movd | count 2
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+sse2 | grep movq | count 3
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep xor

define <4 x i32> @t1(i32 %a) nounwind  {
entry:
        %tmp = insertelement <4 x i32> undef, i32 %a, i32 0
	%tmp6 = shufflevector <4 x i32> zeroinitializer, <4 x i32> %tmp, <4 x i32> < i32 4, i32 1, i32 2, i32 3 >		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp6
}

define <2 x i64> @t2(i64 %a) nounwind  {
entry:
        %tmp = insertelement <2 x i64> undef, i64 %a, i32 0
	%tmp6 = shufflevector <2 x i64> zeroinitializer, <2 x i64> %tmp, <2 x i32> < i32 2, i32 1 >		; <<4 x i32>> [#uses=1]
	ret <2 x i64> %tmp6
}

define <2 x i64> @t3(<2 x i64>* %a) nounwind  {
entry:
	%tmp4 = load <2 x i64>* %a, align 16		; <<2 x i64>> [#uses=1]
	%tmp6 = bitcast <2 x i64> %tmp4 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp7 = shufflevector <4 x i32> zeroinitializer, <4 x i32> %tmp6, <4 x i32> < i32 4, i32 5, i32 2, i32 3 >		; <<4 x i32>> [#uses=1]
	%tmp8 = bitcast <4 x i32> %tmp7 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp8
}

define <2 x i64> @t4(<2 x i64> %a) nounwind  {
entry:
	%tmp5 = bitcast <2 x i64> %a to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp6 = shufflevector <4 x i32> zeroinitializer, <4 x i32> %tmp5, <4 x i32> < i32 4, i32 5, i32 2, i32 3 >		; <<4 x i32>> [#uses=1]
	%tmp7 = bitcast <4 x i32> %tmp6 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp7
}

define <2 x i64> @t5(<2 x i64> %a) nounwind  {
entry:
	%tmp6 = shufflevector <2 x i64> zeroinitializer, <2 x i64> %a, <2 x i32> < i32 2, i32 1 >		; <<4 x i32>> [#uses=1]
	ret <2 x i64> %tmp6
}
