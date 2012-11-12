; RUN: llc < %s -mtriple=i686-apple-darwin -mattr=+sse2 -relocation-model=pic | grep psllw | grep pb

define void @f() nounwind  {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%tmp4403 = tail call <8 x i16> @llvm.x86.sse2.psubs.w( <8 x i16> zeroinitializer, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=2]
	%tmp4443 = tail call <8 x i16> @llvm.x86.sse2.padds.w( <8 x i16> zeroinitializer, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp4609 = tail call <8 x i16> @llvm.x86.sse2.psll.w( <8 x i16> zeroinitializer, <8 x i16> bitcast (<4 x i32> < i32 3, i32 5, i32 6, i32 9 > to <8 x i16>) )		; <<8 x i16>> [#uses=1]
	%tmp4651 = add <8 x i16> %tmp4609, < i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1 >		; <<8 x i16>> [#uses=1]
	%tmp4658 = tail call <8 x i16> @llvm.x86.sse2.psll.w( <8 x i16> %tmp4651, <8 x i16> bitcast (<4 x i32> < i32 4, i32 1, i32 2, i32 3 > to <8 x i16>) )		; <<8 x i16>> [#uses=1]
	%tmp4669 = tail call <8 x i16> @llvm.x86.sse2.pavg.w( <8 x i16> < i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170 >, <8 x i16> %tmp4443 ) nounwind readnone 		; <<8 x i16>> [#uses=2]
	%tmp4679 = tail call <8 x i16> @llvm.x86.sse2.padds.w( <8 x i16> %tmp4669, <8 x i16> %tmp4669 ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp4689 = add <8 x i16> %tmp4679, %tmp4658		; <<8 x i16>> [#uses=1]
	%tmp4700 = tail call <8 x i16> @llvm.x86.sse2.padds.w( <8 x i16> %tmp4689, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp4708 = bitcast <8 x i16> %tmp4700 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp4772 = add <8 x i16> zeroinitializer, < i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1 >		; <<8 x i16>> [#uses=1]
	%tmp4779 = tail call <8 x i16> @llvm.x86.sse2.psll.w( <8 x i16> %tmp4772, <8 x i16> bitcast (<4 x i32> < i32 3, i32 5, i32 undef, i32 7 > to <8 x i16>) )		; <<8 x i16>> [#uses=1]
	%tmp4810 = add <8 x i16> zeroinitializer, %tmp4779		; <<8 x i16>> [#uses=1]
	%tmp4821 = tail call <8 x i16> @llvm.x86.sse2.padds.w( <8 x i16> %tmp4810, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp4829 = bitcast <8 x i16> %tmp4821 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp4900 = tail call <8 x i16> @llvm.x86.sse2.psll.w( <8 x i16> zeroinitializer, <8 x i16> bitcast (<4 x i32> < i32 1, i32 1, i32 2, i32 2 > to <8 x i16>) )		; <<8 x i16>> [#uses=1]
	%tmp4911 = tail call <8 x i16> @llvm.x86.sse2.pavg.w( <8 x i16> < i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170, i16 -23170 >, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=2]
	%tmp4921 = tail call <8 x i16> @llvm.x86.sse2.padds.w( <8 x i16> %tmp4911, <8 x i16> %tmp4911 ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp4931 = add <8 x i16> %tmp4921, %tmp4900		; <<8 x i16>> [#uses=1]
	%tmp4942 = tail call <8 x i16> @llvm.x86.sse2.padds.w( <8 x i16> %tmp4931, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp4950 = bitcast <8 x i16> %tmp4942 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp4957 = tail call <8 x i16> @llvm.x86.sse2.padds.w( <8 x i16> %tmp4403, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp4958 = bitcast <8 x i16> %tmp4957 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%tmp4967 = tail call <8 x i16> @llvm.x86.sse2.psubs.w( <8 x i16> %tmp4403, <8 x i16> zeroinitializer ) nounwind readnone 		; <<8 x i16>> [#uses=1]
	%tmp4968 = bitcast <8 x i16> %tmp4967 to <2 x i64>		; <<2 x i64>> [#uses=1]
	store <2 x i64> %tmp4829, <2 x i64>* null, align 16
	store <2 x i64> %tmp4958, <2 x i64>* null, align 16
	store <2 x i64> %tmp4968, <2 x i64>* null, align 16
	store <2 x i64> %tmp4950, <2 x i64>* null, align 16
	store <2 x i64> %tmp4708, <2 x i64>* null, align 16
	br label %bb
}

declare <8 x i16> @llvm.x86.sse2.psll.w(<8 x i16>, <8 x i16>) nounwind readnone 

declare <8 x i16> @llvm.x86.sse2.pavg.w(<8 x i16>, <8 x i16>) nounwind readnone 

declare <8 x i16> @llvm.x86.sse2.padds.w(<8 x i16>, <8 x i16>) nounwind readnone 

declare <8 x i16> @llvm.x86.sse2.psubs.w(<8 x i16>, <8 x i16>) nounwind readnone 
