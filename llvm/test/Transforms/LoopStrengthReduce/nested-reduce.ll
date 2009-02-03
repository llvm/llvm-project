; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | not grep mul

; Make sure we don't get a multiply by 6 in this loop.

define i32 @foo(i32 %A, i32 %B, i32 %C, i32 %D) {
entry:
	%tmp.5 = icmp sgt i32 %C, 0		; <i1> [#uses=1]
	%tmp.25 = and i32 %A, 1		; <i32> [#uses=1]
	br label %loopentry.1
loopentry.1:		; preds = %loopexit.1, %entry
	%indvar20 = phi i32 [ 0, %entry ], [ %indvar.next21, %loopexit.1 ]		; <i32> [#uses=2]
	%k.1 = phi i32 [ 0, %entry ], [ %k.1.3, %loopexit.1 ]		; <i32> [#uses=2]
	br i1 %tmp.5, label %no_exit.1.preheader, label %loopexit.1
no_exit.1.preheader:		; preds = %loopentry.1
	%i.0.0 = bitcast i32 %indvar20 to i32		; <i32> [#uses=1]
	%tmp.9 = mul i32 %i.0.0, 6		; <i32> [#uses=1]
	br label %no_exit.1.outer
no_exit.1.outer:		; preds = %cond_true, %no_exit.1.preheader
	%k.1.2.ph = phi i32 [ %k.1, %no_exit.1.preheader ], [ %k.09, %cond_true ]		; <i32> [#uses=2]
	%j.1.2.ph = phi i32 [ 0, %no_exit.1.preheader ], [ %inc.1, %cond_true ]		; <i32> [#uses=1]
	br label %no_exit.1
no_exit.1:		; preds = %cond_continue, %no_exit.1.outer
	%indvar.ui = phi i32 [ 0, %no_exit.1.outer ], [ %indvar.next, %cond_continue ]		; <i32> [#uses=2]
	%indvar = bitcast i32 %indvar.ui to i32		; <i32> [#uses=1]
	%j.1.2 = add i32 %indvar, %j.1.2.ph		; <i32> [#uses=2]
	%tmp.11 = add i32 %j.1.2, %tmp.9		; <i32> [#uses=1]
	%tmp.12 = trunc i32 %tmp.11 to i8		; <i8> [#uses=1]
	%shift.upgrd.1 = zext i8 %tmp.12 to i32		; <i32> [#uses=1]
	%tmp.13 = shl i32 %D, %shift.upgrd.1		; <i32> [#uses=2]
	%tmp.15 = icmp eq i32 %tmp.13, %B		; <i1> [#uses=1]
	%inc.1 = add i32 %j.1.2, 1		; <i32> [#uses=3]
	br i1 %tmp.15, label %cond_true, label %cond_continue
cond_true:		; preds = %no_exit.1
	%tmp.26 = and i32 %tmp.25, %tmp.13		; <i32> [#uses=1]
	%k.09 = add i32 %tmp.26, %k.1.2.ph		; <i32> [#uses=2]
	%tmp.517 = icmp slt i32 %inc.1, %C		; <i1> [#uses=1]
	br i1 %tmp.517, label %no_exit.1.outer, label %loopexit.1
cond_continue:		; preds = %no_exit.1
	%tmp.519 = icmp slt i32 %inc.1, %C		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar.ui, 1		; <i32> [#uses=1]
	br i1 %tmp.519, label %no_exit.1, label %loopexit.1
loopexit.1:		; preds = %cond_continue, %cond_true, %loopentry.1
	%k.1.3 = phi i32 [ %k.1, %loopentry.1 ], [ %k.09, %cond_true ], [ %k.1.2.ph, %cond_continue ]		; <i32> [#uses=2]
	%indvar.next21 = add i32 %indvar20, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next21, 4		; <i1> [#uses=1]
	br i1 %exitcond, label %loopexit.0, label %loopentry.1
loopexit.0:		; preds = %loopexit.1
	ret i32 %k.1.3
}
