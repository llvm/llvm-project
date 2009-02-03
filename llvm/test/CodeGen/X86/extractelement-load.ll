; RUN: llvm-as %s -o - | llc -march=x86 -mattr=+sse2 -mcpu=yonah | not grep movd
; RUN: llvm-as %s -o - | llc -march=x86-64 -mattr=+sse2 -mcpu=core2 | not grep movd

define i32 @t(<2 x i64>* %val) nounwind  {
	%tmp2 = load <2 x i64>* %val, align 16		; <<2 x i64>> [#uses=1]
	%tmp3 = bitcast <2 x i64> %tmp2 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp4 = extractelement <4 x i32> %tmp3, i32 2		; <i32> [#uses=1]
	ret i32 %tmp4
}
