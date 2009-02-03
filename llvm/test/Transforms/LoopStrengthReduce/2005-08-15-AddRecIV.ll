; RUN: llvm-as < %s | opt -loop-reduce -disable-output

define void @try_swap() {
entry:
	br i1 false, label %cond_continue.0.i, label %cond_false.0.i
cond_false.0.i:		; preds = %entry
	ret void
cond_continue.0.i:		; preds = %entry
	br i1 false, label %cond_continue.1.i, label %cond_false.1.i
cond_false.1.i:		; preds = %cond_continue.0.i
	ret void
cond_continue.1.i:		; preds = %cond_continue.0.i
	br i1 false, label %endif.3.i, label %else.0.i
endif.3.i:		; preds = %cond_continue.1.i
	br i1 false, label %my_irand.exit82, label %endif.0.i62
else.0.i:		; preds = %cond_continue.1.i
	ret void
endif.0.i62:		; preds = %endif.3.i
	ret void
my_irand.exit82:		; preds = %endif.3.i
	br i1 false, label %else.2, label %then.4
then.4:		; preds = %my_irand.exit82
	ret void
else.2:		; preds = %my_irand.exit82
	br i1 false, label %find_affected_nets.exit, label %loopentry.1.i107.outer.preheader
loopentry.1.i107.outer.preheader:		; preds = %else.2
	ret void
find_affected_nets.exit:		; preds = %else.2
	br i1 false, label %save_region_occ.exit, label %loopentry.1
save_region_occ.exit:		; preds = %find_affected_nets.exit
	br i1 false, label %no_exit.1.preheader, label %loopexit.1
loopentry.1:		; preds = %find_affected_nets.exit
	ret void
no_exit.1.preheader:		; preds = %save_region_occ.exit
	ret void
loopexit.1:		; preds = %save_region_occ.exit
	br i1 false, label %then.10, label %loopentry.3
then.10:		; preds = %loopexit.1
	ret void
loopentry.3:		; preds = %endif.16, %loopexit.1
	%indvar342 = phi i32 [ %indvar.next343, %endif.16 ], [ 0, %loopexit.1 ]		; <i32> [#uses=2]
	br i1 false, label %loopexit.3, label %endif.16
endif.16:		; preds = %loopentry.3
	%indvar.next343 = add i32 %indvar342, 1		; <i32> [#uses=1]
	br label %loopentry.3
loopexit.3:		; preds = %loopentry.3
	br label %loopentry.4
loopentry.4:		; preds = %loopentry.4, %loopexit.3
	%indvar340 = phi i32 [ 0, %loopexit.3 ], [ %indvar.next341, %loopentry.4 ]		; <i32> [#uses=2]
	%tmp. = add i32 %indvar340, %indvar342		; <i32> [#uses=1]
	%tmp.526 = load i32** null		; <i32*> [#uses=1]
	%gep.upgrd.1 = zext i32 %tmp. to i64		; <i64> [#uses=1]
	%tmp.528 = getelementptr i32* %tmp.526, i64 %gep.upgrd.1		; <i32*> [#uses=1]
	store i32 0, i32* %tmp.528
	%indvar.next341 = add i32 %indvar340, 1		; <i32> [#uses=1]
	br label %loopentry.4
}
