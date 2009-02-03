; RUN: llvm-as < %s | llc -march=arm -mattr=+v6,+vfp2

@quant_coef = external global [6 x [4 x [4 x i32]]]		; <[6 x [4 x [4 x i32]]]*> [#uses=1]
@dequant_coef = external global [6 x [4 x [4 x i32]]]		; <[6 x [4 x [4 x i32]]]*> [#uses=1]
@A = external global [4 x [4 x i32]]		; <[4 x [4 x i32]]*> [#uses=1]

define fastcc i32 @dct_luma_sp(i32 %block_x, i32 %block_y, i32* %coeff_cost) {
entry:
	%predicted_block = alloca [4 x [4 x i32]], align 4		; <[4 x [4 x i32]]*> [#uses=1]
	br label %cond_next489

cond_next489:		; preds = %cond_false, %bb471
	%j.7.in = load i8* null		; <i8> [#uses=1]
	%i.8.in = load i8* null		; <i8> [#uses=1]
	%i.8 = zext i8 %i.8.in to i32		; <i32> [#uses=4]
	%j.7 = zext i8 %j.7.in to i32		; <i32> [#uses=4]
	%tmp495 = getelementptr [4 x [4 x i32]]* %predicted_block, i32 0, i32 %i.8, i32 %j.7		; <i32*> [#uses=2]
	%tmp496 = load i32* %tmp495		; <i32> [#uses=2]
	%tmp502 = load i32* null		; <i32> [#uses=1]
	%tmp542 = getelementptr [6 x [4 x [4 x i32]]]* @quant_coef, i32 0, i32 0, i32 %i.8, i32 %j.7		; <i32*> [#uses=1]
	%tmp543 = load i32* %tmp542		; <i32> [#uses=1]
	%tmp548 = ashr i32 0, 0		; <i32> [#uses=3]
	%tmp561 = sub i32 0, %tmp496		; <i32> [#uses=3]
	%abscond563 = icmp sgt i32 %tmp561, -1		; <i1> [#uses=1]
	%abs564 = select i1 %abscond563, i32 %tmp561, i32 0		; <i32> [#uses=1]
	%tmp572 = mul i32 %abs564, %tmp543		; <i32> [#uses=1]
	%tmp574 = add i32 %tmp572, 0		; <i32> [#uses=1]
	%tmp576 = ashr i32 %tmp574, 0		; <i32> [#uses=7]
	%tmp579 = icmp eq i32 %tmp548, %tmp576		; <i1> [#uses=1]
	br i1 %tmp579, label %bb712, label %cond_next589

cond_next589:		; preds = %cond_next489
	%tmp605 = getelementptr [6 x [4 x [4 x i32]]]* @dequant_coef, i32 0, i32 0, i32 %i.8, i32 %j.7		; <i32*> [#uses=1]
	%tmp606 = load i32* %tmp605		; <i32> [#uses=1]
	%tmp612 = load i32* null		; <i32> [#uses=1]
	%tmp629 = load i32* null		; <i32> [#uses=1]
	%tmp629a = sitofp i32 %tmp629 to double		; <double> [#uses=1]
	%tmp631 = mul double %tmp629a, 0.000000e+00		; <double> [#uses=1]
	%tmp632 = add double 0.000000e+00, %tmp631		; <double> [#uses=1]
	%tmp642 = call fastcc i32 @sign( i32 %tmp576, i32 %tmp561 )		; <i32> [#uses=1]
	%tmp650 = mul i32 %tmp606, %tmp642		; <i32> [#uses=1]
	%tmp656 = mul i32 %tmp650, %tmp612		; <i32> [#uses=1]
	%tmp658 = shl i32 %tmp656, 0		; <i32> [#uses=1]
	%tmp659 = ashr i32 %tmp658, 6		; <i32> [#uses=1]
	%tmp660 = sub i32 0, %tmp659		; <i32> [#uses=1]
	%tmp666 = sub i32 %tmp660, %tmp496		; <i32> [#uses=1]
	%tmp667 = sitofp i32 %tmp666 to double		; <double> [#uses=2]
	call void @levrun_linfo_inter( i32 %tmp576, i32 0, i32* null, i32* null )
	%tmp671 = mul double %tmp667, %tmp667		; <double> [#uses=1]
	%tmp675 = add double %tmp671, 0.000000e+00		; <double> [#uses=1]
	%tmp678 = fcmp oeq double %tmp632, %tmp675		; <i1> [#uses=1]
	br i1 %tmp678, label %cond_true679, label %cond_false693

cond_true679:		; preds = %cond_next589
	%abscond681 = icmp sgt i32 %tmp548, -1		; <i1> [#uses=1]
	%abs682 = select i1 %abscond681, i32 %tmp548, i32 0		; <i32> [#uses=1]
	%abscond684 = icmp sgt i32 %tmp576, -1		; <i1> [#uses=1]
	%abs685 = select i1 %abscond684, i32 %tmp576, i32 0		; <i32> [#uses=1]
	%tmp686 = icmp slt i32 %abs682, %abs685		; <i1> [#uses=1]
	br i1 %tmp686, label %cond_next702, label %cond_false689

cond_false689:		; preds = %cond_true679
	%tmp739 = icmp eq i32 %tmp576, 0		; <i1> [#uses=1]
	br i1 %tmp579, label %bb737, label %cond_false708

cond_false693:		; preds = %cond_next589
	ret i32 0

cond_next702:		; preds = %cond_true679
	ret i32 0

cond_false708:		; preds = %cond_false689
	ret i32 0

bb712:		; preds = %cond_next489
	ret i32 0

bb737:		; preds = %cond_false689
	br i1 %tmp739, label %cond_next791, label %cond_true740

cond_true740:		; preds = %bb737
	%tmp761 = call fastcc i32 @sign( i32 %tmp576, i32 0 )		; <i32> [#uses=1]
	%tmp780 = load i32* null		; <i32> [#uses=1]
	%tmp785 = getelementptr [4 x [4 x i32]]* @A, i32 0, i32 %i.8, i32 %j.7		; <i32*> [#uses=1]
	%tmp786 = load i32* %tmp785		; <i32> [#uses=1]
	%tmp781 = mul i32 %tmp780, %tmp761		; <i32> [#uses=1]
	%tmp787 = mul i32 %tmp781, %tmp786		; <i32> [#uses=1]
	%tmp789 = shl i32 %tmp787, 0		; <i32> [#uses=1]
	%tmp790 = ashr i32 %tmp789, 6		; <i32> [#uses=1]
	br label %cond_next791

cond_next791:		; preds = %cond_true740, %bb737
	%ilev.1 = phi i32 [ %tmp790, %cond_true740 ], [ 0, %bb737 ]		; <i32> [#uses=1]
	%tmp796 = load i32* %tmp495		; <i32> [#uses=1]
	%tmp798 = add i32 %tmp796, %ilev.1		; <i32> [#uses=1]
	%tmp812 = mul i32 0, %tmp502		; <i32> [#uses=0]
	%tmp818 = call fastcc i32 @sign( i32 0, i32 %tmp798 )		; <i32> [#uses=0]
	unreachable
}

declare i32 @sign(i32, i32)

declare void @levrun_linfo_inter(i32, i32, i32*, i32*)
