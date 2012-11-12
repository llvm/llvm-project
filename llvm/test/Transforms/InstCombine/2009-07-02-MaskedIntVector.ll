; RUN: opt < %s -instcombine | llvm-dis
; PR4495

define i32 @test(i64 %test) {
entry:
	%0 = bitcast <4 x i32> undef to <16 x i8>		; <<16 x i8>> [#uses=1]
	%t12 = shufflevector <16 x i8> %0, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 16, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>		; <<16 x i8>> [#uses=1]
	%t11 = bitcast <16 x i8> %t12 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%t9 = extractelement <2 x i64> %t11, i32 0		; <i64> [#uses=1]
	%t10 = bitcast i64 %t9 to <2 x i32>		; <<2 x i32>> [#uses=1]
	%t7 = bitcast i64 %test to <2 x i32>		; <<2 x i32>> [#uses=1]
	%t6 = xor <2 x i32> %t10, %t7		; <<2 x i32>> [#uses=1]
	%t1 = extractelement <2 x i32> %t6, i32 0		; <i32> [#uses=1]
	ret i32 %t1
}
