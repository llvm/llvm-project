; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx | grep movd
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx | grep movq

define void @foo(<1 x i64>* %a, <1 x i64>* %b) nounwind {
entry:
	%0 = load <1 x i64>* %a, align 8		; <<1 x i64>> [#uses=1]
	%1 = bitcast <1 x i64> %0 to <2 x i32>		; <<2 x i32>> [#uses=1]
	%2 = and <2 x i32> %1, < i32 -1, i32 0 >		; <<2 x i32>> [#uses=1]
	%3 = bitcast <2 x i32> %2 to <1 x i64>		; <<1 x i64>> [#uses=1]
	store <1 x i64> %3, <1 x i64>* %b, align 8
	br label %bb2

bb2:		; preds = %entry
	ret void
}
