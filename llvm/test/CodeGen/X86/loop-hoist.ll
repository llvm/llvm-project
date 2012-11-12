; LSR should hoist the load from the "Arr" stub out of the loop.

; RUN: llc < %s -relocation-model=dynamic-no-pic -mtriple=i686-apple-darwin8.7.2 | FileCheck %s

; CHECK: _foo:
; CHECK:    L_Arr$non_lazy_ptr
; CHECK: LBB0_1:

@Arr = external global [0 x i32]		; <[0 x i32]*> [#uses=1]

define void @foo(i32 %N.in, i32 %x) nounwind {
entry:
	%N = bitcast i32 %N.in to i32		; <i32> [#uses=1]
	br label %cond_true

cond_true:		; preds = %cond_true, %entry
	%indvar = phi i32 [ %x, %entry ], [ %indvar.next, %cond_true ]		; <i32> [#uses=2]
	%i.0.0 = bitcast i32 %indvar to i32		; <i32> [#uses=2]
	%tmp = getelementptr [0 x i32]* @Arr, i32 0, i32 %i.0.0		; <i32*> [#uses=1]
	store i32 %i.0.0, i32* %tmp
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %N		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %cond_true

return:		; preds = %cond_true
	ret void
}
