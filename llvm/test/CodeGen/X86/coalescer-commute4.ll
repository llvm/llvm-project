; RUN: llc < %s -mtriple=i686-apple-darwin -mattr=+sse2 | not grep movaps
; PR1501

define float @foo(i32* %x, float* %y, i32 %c) nounwind  {
entry:
	%tmp2132 = icmp eq i32 %c, 0		; <i1> [#uses=2]
	br i1 %tmp2132, label %bb23, label %bb.preheader

bb.preheader:		; preds = %entry
	%umax = select i1 %tmp2132, i32 1, i32 %c		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.preheader
	%i.0.reg2mem.0 = phi i32 [ 0, %bb.preheader ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	%res.0.reg2mem.0 = phi float [ 0.000000e+00, %bb.preheader ], [ %tmp14, %bb ]		; <float> [#uses=1]
	%tmp3 = getelementptr i32* %x, i32 %i.0.reg2mem.0		; <i32*> [#uses=1]
	%tmp4 = load i32* %tmp3, align 4		; <i32> [#uses=1]
	%tmp45 = sitofp i32 %tmp4 to float		; <float> [#uses=1]
	%tmp8 = getelementptr float* %y, i32 %i.0.reg2mem.0		; <float*> [#uses=1]
	%tmp9 = load float* %tmp8, align 4		; <float> [#uses=1]
	%tmp11 = fmul float %tmp9, %tmp45		; <float> [#uses=1]
	%tmp14 = fadd float %tmp11, %res.0.reg2mem.0		; <float> [#uses=2]
	%indvar.next = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %bb23, label %bb

bb23:		; preds = %bb, %entry
	%res.0.reg2mem.1 = phi float [ 0.000000e+00, %entry ], [ %tmp14, %bb ]		; <float> [#uses=1]
	ret float %res.0.reg2mem.1
}
