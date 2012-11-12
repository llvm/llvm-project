; RUN: llc < %s -march=ppc32 -mcpu=g5
; END.

define void @test(i8* %stack) {
entry:
	%tmp9 = icmp eq i32 0, 0		; <i1> [#uses=1]
	%tmp30 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp30, label %cond_next54, label %cond_true31
cond_true860:		; preds = %bb855
	%tmp879 = tail call <4 x float> @llvm.ppc.altivec.vmaddfp( <4 x float> zeroinitializer, <4 x float> zeroinitializer, <4 x float> zeroinitializer )		; <<4 x float>> [#uses=1]
	%tmp880 = bitcast <4 x float> %tmp879 to <4 x i32>		; <<4 x i32>> [#uses=2]
	%tmp883 = shufflevector <4 x i32> %tmp880, <4 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x i32>> [#uses=1]
	%tmp883.upgrd.1 = bitcast <4 x i32> %tmp883 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp885 = shufflevector <4 x i32> %tmp880, <4 x i32> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x i32>> [#uses=1]
	%tmp885.upgrd.2 = bitcast <4 x i32> %tmp885 to <4 x float>		; <<4 x float>> [#uses=1]
	br label %cond_next905
cond_true31:		; preds = %entry
	ret void
cond_next54:		; preds = %entry
	br i1 %tmp9, label %cond_false385, label %bb279
bb279:		; preds = %cond_next54
	ret void
cond_false385:		; preds = %cond_next54
	%tmp388 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp388, label %cond_next463, label %cond_true389
cond_true389:		; preds = %cond_false385
	ret void
cond_next463:		; preds = %cond_false385
	%tmp1208107 = icmp ugt i8* null, %stack		; <i1> [#uses=1]
	br i1 %tmp1208107, label %cond_true1209.preheader, label %bb1212
cond_true498:		; preds = %cond_true1209.preheader
	ret void
cond_true519:		; preds = %cond_true1209.preheader
	%bothcond = or i1 false, false		; <i1> [#uses=1]
	br i1 %bothcond, label %bb855, label %bb980
cond_false548:		; preds = %cond_true1209.preheader
	ret void
bb855:		; preds = %cond_true519
	%tmp859 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp859, label %cond_true860, label %cond_next905
cond_next905:		; preds = %bb855, %cond_true860
	%vfpw2.4 = phi <4 x float> [ %tmp885.upgrd.2, %cond_true860 ], [ undef, %bb855 ]		; <<4 x float>> [#uses=0]
	%vfpw1.4 = phi <4 x float> [ %tmp883.upgrd.1, %cond_true860 ], [ undef, %bb855 ]		; <<4 x float>> [#uses=0]
	%tmp930 = bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>> [#uses=0]
	ret void
bb980:		; preds = %cond_true519
	ret void
cond_true1209.preheader:		; preds = %cond_next463
	%tmp496 = and i32 0, 12288		; <i32> [#uses=1]
	switch i32 %tmp496, label %cond_false548 [
		 i32 0, label %cond_true498
		 i32 4096, label %cond_true519
	]
bb1212:		; preds = %cond_next463
	ret void
}

declare <4 x float> @llvm.ppc.altivec.vmaddfp(<4 x float>, <4 x float>, <4 x float>)
