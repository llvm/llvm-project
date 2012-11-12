; RUN: llc < %s -mtriple=i686-apple-darwin -mattr=+sse2 | not grep movaps

define i32 @t() {
entry:
	br i1 true, label %bb1664, label %bb1656
bb1656:		; preds = %entry
	ret i32 0
bb1664:		; preds = %entry
	%tmp4297 = bitcast <16 x i8> zeroinitializer to <2 x i64>		; <<2 x i64>> [#uses=2]
	%tmp4351 = call <16 x i8> @llvm.x86.sse2.pcmpeq.b( <16 x i8> zeroinitializer, <16 x i8> zeroinitializer ) nounwind readnone 		; <<16 x i8>> [#uses=0]
	br i1 false, label %bb5310, label %bb4743
bb4743:		; preds = %bb1664
	%tmp4360.not28 = or <2 x i64> zeroinitializer, %tmp4297		; <<2 x i64>> [#uses=1]
	br label %bb5310
bb5310:		; preds = %bb4743, %bb1664
	%tmp4360.not28.pn = phi <2 x i64> [ %tmp4360.not28, %bb4743 ], [ %tmp4297, %bb1664 ]		; <<2 x i64>> [#uses=1]
	%tmp4415.not.pn = or <2 x i64> zeroinitializer, %tmp4360.not28.pn		; <<2 x i64>> [#uses=0]
	ret i32 0
}

declare <16 x i8> @llvm.x86.sse2.pcmpeq.b(<16 x i8>, <16 x i8>) nounwind readnone 
