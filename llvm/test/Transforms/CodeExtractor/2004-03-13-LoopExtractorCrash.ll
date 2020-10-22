; RUN: opt < %s -loop-simplify -loop-extract -disable-output

define void @solve() {
entry:
	br label %loopentry.0

loopentry.0:		; preds = %endif.0, %entry
	br i1 false, label %no_exit.0, label %loopexit.0

no_exit.0:		; preds = %loopentry.0
	br i1 false, label %then.0, label %endif.0

then.0:		; preds = %no_exit.0
	br i1 false, label %shortcirc_done, label %shortcirc_next

shortcirc_next:		; preds = %then.0
	br label %shortcirc_done

shortcirc_done:		; preds = %shortcirc_next, %then.0
	br i1 false, label %then.1, label %endif.1

then.1:		; preds = %shortcirc_done
	br i1 false, label %cond_true, label %cond_false

cond_true:		; preds = %then.1
	br label %cond_continue

cond_false:		; preds = %then.1
	br label %cond_continue

cond_continue:		; preds = %cond_false, %cond_true
	br label %return

after_ret.0:		; No predecessors!
	br label %endif.1

endif.1:		; preds = %after_ret.0, %shortcirc_done
	br label %endif.0

endif.0:		; preds = %endif.1, %no_exit.0
	br label %loopentry.0

loopexit.0:		; preds = %loopentry.0
	br i1 false, label %then.2, label %endif.2

then.2:		; preds = %loopexit.0
	br i1 false, label %then.3, label %endif.3

then.3:		; preds = %then.2
	br label %return

after_ret.1:		; No predecessors!
	br label %endif.3

endif.3:		; preds = %after_ret.1, %then.2
	br label %endif.2

endif.2:		; preds = %endif.3, %loopexit.0
	br label %loopentry.1

loopentry.1:		; preds = %no_exit.1, %endif.2
	br i1 false, label %no_exit.1, label %loopexit.1

no_exit.1:		; preds = %loopentry.1
	br label %loopentry.1

loopexit.1:		; preds = %loopentry.1
	br label %return

after_ret.2:		; No predecessors!
	br label %return

return:		; preds = %after_ret.2, %loopexit.1, %then.3, %cond_continue
	ret void
}
