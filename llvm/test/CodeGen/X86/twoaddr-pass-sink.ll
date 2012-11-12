; RUN: llc < %s -march=x86 -mattr=+sse2 -stats 2>&1 | grep "Number of 3-address instructions sunk"

define void @t2(<2 x i64>* %vDct, <2 x i64>* %vYp, i8* %skiplist, <2 x i64> %a1) nounwind  {
entry:
	%tmp25 = bitcast <2 x i64> %a1 to <8 x i16>		; <<8 x i16>> [#uses=1]
	br label %bb
bb:		; preds = %bb, %entry
	%skiplist_addr.0.rec = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	%vYp_addr.0.rec = shl i32 %skiplist_addr.0.rec, 3		; <i32> [#uses=3]
	%vDct_addr.0 = getelementptr <2 x i64>* %vDct, i32 %vYp_addr.0.rec		; <<2 x i64>*> [#uses=1]
	%vYp_addr.0 = getelementptr <2 x i64>* %vYp, i32 %vYp_addr.0.rec		; <<2 x i64>*> [#uses=1]
	%skiplist_addr.0 = getelementptr i8* %skiplist, i32 %skiplist_addr.0.rec		; <i8*> [#uses=1]
	%vDct_addr.0.sum43 = or i32 %vYp_addr.0.rec, 1		; <i32> [#uses=1]
	%tmp7 = getelementptr <2 x i64>* %vDct, i32 %vDct_addr.0.sum43		; <<2 x i64>*> [#uses=1]
	%tmp8 = load <2 x i64>* %tmp7, align 16		; <<2 x i64>> [#uses=1]
	%tmp11 = load <2 x i64>* %vDct_addr.0, align 16		; <<2 x i64>> [#uses=1]
	%tmp13 = bitcast <2 x i64> %tmp8 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp15 = bitcast <2 x i64> %tmp11 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp16 = shufflevector <8 x i16> %tmp15, <8 x i16> %tmp13, <8 x i32> < i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11 >		; <<8 x i16>> [#uses=1]
	%tmp26 = mul <8 x i16> %tmp25, %tmp16		; <<8 x i16>> [#uses=1]
	%tmp27 = bitcast <8 x i16> %tmp26 to <2 x i64>		; <<2 x i64>> [#uses=1]
	store <2 x i64> %tmp27, <2 x i64>* %vYp_addr.0, align 16
	%tmp37 = load i8* %skiplist_addr.0, align 1		; <i8> [#uses=1]
	%tmp38 = icmp eq i8 %tmp37, 0		; <i1> [#uses=1]
	%indvar.next = add i32 %skiplist_addr.0.rec, 1		; <i32> [#uses=1]
	br i1 %tmp38, label %return, label %bb
return:		; preds = %bb
	ret void
}
