; RUN: llc < %s -march=x86 -mcpu=generic -mattr=+sse2 | not grep lea

define float @foo(i32* %x, float* %y, i32 %c) nounwind {
entry:
	%tmp2132 = icmp eq i32 %c, 0		; <i1> [#uses=1]
	br i1 %tmp2132, label %bb23, label %bb18

bb18:		; preds = %bb18, %entry
	%i.0.reg2mem.0 = phi i32 [ 0, %entry ], [ %tmp17, %bb18 ]		; <i32> [#uses=3]
	%res.0.reg2mem.0 = phi float [ 0.000000e+00, %entry ], [ %tmp14, %bb18 ]		; <float> [#uses=1]
	%tmp3 = getelementptr i32, i32* %x, i32 %i.0.reg2mem.0		; <i32*> [#uses=1]
	%tmp4 = load i32, i32* %tmp3, align 4		; <i32> [#uses=1]
	%tmp45 = sitofp i32 %tmp4 to float		; <float> [#uses=1]
	%tmp8 = getelementptr float, float* %y, i32 %i.0.reg2mem.0		; <float*> [#uses=1]
	%tmp9 = load float, float* %tmp8, align 4		; <float> [#uses=1]
	%tmp11 = fmul float %tmp9, %tmp45		; <float> [#uses=1]
	%tmp14 = fadd float %tmp11, %res.0.reg2mem.0		; <float> [#uses=2]
	%tmp17 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%tmp21 = icmp ult i32 %tmp17, %c		; <i1> [#uses=1]
	br i1 %tmp21, label %bb18, label %bb23

bb23:		; preds = %bb18, %entry
	%res.0.reg2mem.1 = phi float [ 0.000000e+00, %entry ], [ %tmp14, %bb18 ]		; <float> [#uses=1]
	ret float %res.0.reg2mem.1
}
