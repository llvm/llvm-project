
; Update loop iteraton space to eliminate condition inside loop.
; RUN: llvm-as < %s | opt -loop-index-split | llvm-dis | not grep bothcond
define void @test(float* %x, i32 %ndat, float** %y, float %xcen, i32 %xmin, i32 %xmax, float %sigmal, float %contribution) {
entry:
	%tmp519 = icmp sgt i32 %xmin, %xmax		; <i1> [#uses=1]
	br i1 %tmp519, label %return, label %bb.preheader

bb.preheader:		; preds = %entry
	%tmp3031 = fpext float %contribution to double		; <double> [#uses=1]
	%tmp32 = mul double %tmp3031, 5.000000e-01		; <double> [#uses=1]
	%tmp3839 = fpext float %sigmal to double		; <double> [#uses=1]
	br label %bb

bb:		; preds = %bb.preheader, %cond_next45
	%i.01.0 = phi i32 [ %tmp47, %cond_next45 ], [ 0, %bb.preheader ]		; <i32> [#uses=6]
	%tmp2 = icmp sgt i32 %i.01.0, -1		; <i1> [#uses=1]
	%tmp6 = icmp slt i32 %i.01.0, %ndat		; <i1> [#uses=1]
	%bothcond = and i1 %tmp2, %tmp6		; <i1> [#uses=1]
	br i1 %bothcond, label %cond_true9, label %cond_next45

cond_true9:		; preds = %bb
	%tmp12 = getelementptr float* %x, i32 %i.01.0		; <float*> [#uses=1]
	%tmp13 = load float* %tmp12, align 4		; <float> [#uses=1]
	%tmp15 = sub float %xcen, %tmp13		; <float> [#uses=1]
	%tmp16 = tail call float @fabsf( float %tmp15 )		; <float> [#uses=1]
	%tmp18 = fdiv float %tmp16, %sigmal		; <float> [#uses=1]
	%tmp21 = load float** %y, align 4		; <float*> [#uses=2]
	%tmp27 = getelementptr float* %tmp21, i32 %i.01.0		; <float*> [#uses=1]
	%tmp28 = load float* %tmp27, align 4		; <float> [#uses=1]
	%tmp2829 = fpext float %tmp28 to double		; <double> [#uses=1]
	%tmp34 = sub float -0.000000e+00, %tmp18		; <float> [#uses=1]
	%tmp3435 = fpext float %tmp34 to double		; <double> [#uses=1]
	%tmp36 = tail call double @exp( double %tmp3435 )		; <double> [#uses=1]
	%tmp37 = mul double %tmp32, %tmp36		; <double> [#uses=1]
	%tmp40 = fdiv double %tmp37, %tmp3839		; <double> [#uses=1]
	%tmp41 = add double %tmp2829, %tmp40		; <double> [#uses=1]
	%tmp4142 = fptrunc double %tmp41 to float		; <float> [#uses=1]
	%tmp44 = getelementptr float* %tmp21, i32 %i.01.0		; <float*> [#uses=1]
	store float %tmp4142, float* %tmp44, align 4
	br label %cond_next45

cond_next45:		; preds = %bb, %cond_true9
	%tmp47 = add i32 %i.01.0, 1		; <i32> [#uses=2]
	%tmp51 = icmp sgt i32 %tmp47, %xmax		; <i1> [#uses=1]
	br i1 %tmp51, label %return.loopexit, label %bb

return.loopexit:		; preds = %cond_next45
	br label %return

return:		; preds = %return.loopexit, %entry
	ret void
}

declare float @fabsf(float)

declare double @exp(double)
