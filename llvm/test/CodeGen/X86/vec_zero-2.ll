; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2

define i32 @t() {
entry:
	br i1 true, label %bb4743, label %bb1656
bb1656:		; preds = %entry
	ret i32 0
bb1664:		; preds = %entry
	br i1 false, label %bb5310, label %bb4743
bb4743:		; preds = %bb1664
	%tmp5256 = bitcast <2 x i64> zeroinitializer to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp5257 = sub <8 x i16> %tmp5256, zeroinitializer		; <<8 x i16>> [#uses=1]
	%tmp5258 = bitcast <8 x i16> %tmp5257 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp5265 = bitcast <2 x i64> %tmp5258 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp5266 = call <8 x i16> @llvm.x86.sse2.packuswb.128( <8 x i16> %tmp5265, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp5267 = bitcast <8 x i16> %tmp5266 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp5294 = and <2 x i64> zeroinitializer, %tmp5267		; <<2 x i64>> [#uses=1]
	br label %bb5310
bb5310:		; preds = %bb4743, %bb1664
	%tmp5294.pn = phi <2 x i64> [ %tmp5294, %bb4743 ], [ zeroinitializer, %bb1664 ]		; <<2 x i64>> [#uses=0]
	ret i32 0
}

declare <8 x i16> @llvm.x86.sse2.packuswb.128(<8 x i16>, <8 x i16>) nounwind readnone 
