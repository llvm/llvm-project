; RUN: llc < %s -march=x86-64 -mcpu=core2 -o %t
; RUN: not grep unpcklps %t

define i32 @foo() nounwind {
entry:
	%tmp74.i25762 = shufflevector <16 x float> zeroinitializer, <16 x float> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>		; <<16 x float>> [#uses=1]
	%tmp518 = shufflevector <16 x float> %tmp74.i25762, <16 x float> undef, <4 x i32> <i32 12, i32 13, i32 14, i32 15>		; <<4 x float>> [#uses=1]
	%movss.i25611 = shufflevector <4 x float> zeroinitializer, <4 x float> %tmp518, <4 x i32> <i32 4, i32 1, i32 2, i32 3>		; <<4 x float>> [#uses=1]
	%conv3.i25615 = shufflevector <4 x float> %movss.i25611, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>		; <<4 x float>> [#uses=1]
	%sub.i25620 = fsub <4 x float> %conv3.i25615, zeroinitializer		; <<4 x float>> [#uses=1]
	%mul.i25621 = fmul <4 x float> zeroinitializer, %sub.i25620		; <<4 x float>> [#uses=1]
	%add.i25622 = fadd <4 x float> zeroinitializer, %mul.i25621		; <<4 x float>> [#uses=1]
	store <4 x float> %add.i25622, <4 x float>* null
	unreachable
}
