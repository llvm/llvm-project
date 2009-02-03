; RUN: llvm-as < %s | llc -mtriple=i686-apple-darwin -mattr=+sse2 | not grep movaps
; PR1877

@NNTOT = weak global i32 0		; <i32*> [#uses=1]
@G = weak global float 0.000000e+00		; <float*> [#uses=1]

define void @runcont(i32* %source) nounwind  {
entry:
	%tmp10 = load i32* @NNTOT, align 4		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %entry
	%neuron.0 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%thesum.0 = phi float [ 0.000000e+00, %entry ], [ %tmp6, %bb ]		; <float> [#uses=1]
	%tmp2 = getelementptr i32* %source, i32 %neuron.0		; <i32*> [#uses=1]
	%tmp3 = load i32* %tmp2, align 4		; <i32> [#uses=1]
	%tmp34 = sitofp i32 %tmp3 to float		; <float> [#uses=1]
	%tmp6 = add float %tmp34, %thesum.0		; <float> [#uses=2]
	%indvar.next = add i32 %neuron.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %tmp10		; <i1> [#uses=1]
	br i1 %exitcond, label %bb13, label %bb

bb13:		; preds = %bb
	volatile store float %tmp6, float* @G, align 4
	ret void
}
