; RUN: opt < %s -lcssa -disable-output -verify-dom-info -verify-loop-info
; PR977
; END.
declare i32 @opost_block()

define void @write_chan() {
entry:
	br i1 false, label %shortcirc_next.0, label %shortcirc_done.0
shortcirc_next.0:		; preds = %entry
	br label %shortcirc_done.0
shortcirc_done.0:		; preds = %shortcirc_next.0, %entry
	br i1 false, label %shortcirc_next.1, label %shortcirc_done.1
shortcirc_next.1:		; preds = %shortcirc_done.0
	br label %shortcirc_done.1
shortcirc_done.1:		; preds = %shortcirc_next.1, %shortcirc_done.0
	br i1 false, label %then.0, label %endif.0
then.0:		; preds = %shortcirc_done.1
	br i1 false, label %then.1, label %endif.1
then.1:		; preds = %then.0
	br label %return
after_ret.0:		; No predecessors!
	br label %endif.1
endif.1:		; preds = %after_ret.0, %then.0
	br label %endif.0
endif.0:		; preds = %endif.1, %shortcirc_done.1
	br label %loopentry.0
loopentry.0:		; preds = %endif.12, %endif.0
	br i1 false, label %then.2, label %endif.2
then.2:		; preds = %loopentry.0
	br label %loopexit.0
dead_block_after_break.0:		; No predecessors!
	br label %endif.2
endif.2:		; preds = %dead_block_after_break.0, %loopentry.0
	br i1 false, label %shortcirc_done.2, label %shortcirc_next.2
shortcirc_next.2:		; preds = %endif.2
	br i1 false, label %shortcirc_next.3, label %shortcirc_done.3
shortcirc_next.3:		; preds = %shortcirc_next.2
	br label %shortcirc_done.3
shortcirc_done.3:		; preds = %shortcirc_next.3, %shortcirc_next.2
	br label %shortcirc_done.2
shortcirc_done.2:		; preds = %shortcirc_done.3, %endif.2
	br i1 false, label %then.3, label %endif.3
then.3:		; preds = %shortcirc_done.2
	br label %loopexit.0
dead_block_after_break.1:		; No predecessors!
	br label %endif.3
endif.3:		; preds = %dead_block_after_break.1, %shortcirc_done.2
	br i1 false, label %shortcirc_next.4, label %shortcirc_done.4
shortcirc_next.4:		; preds = %endif.3
	br label %shortcirc_done.4
shortcirc_done.4:		; preds = %shortcirc_next.4, %endif.3
	br i1 false, label %then.4, label %else
then.4:		; preds = %shortcirc_done.4
	br label %loopentry.1
loopentry.1:		; preds = %endif.8, %then.4
	br i1 false, label %no_exit, label %loopexit.1
no_exit:		; preds = %loopentry.1
	%tmp.94 = call i32 @opost_block( )		; <i32> [#uses=1]
	br i1 false, label %then.5, label %endif.5
then.5:		; preds = %no_exit
	br i1 false, label %then.6, label %endif.6
then.6:		; preds = %then.5
	br label %loopexit.1
dead_block_after_break.2:		; No predecessors!
	br label %endif.6
endif.6:		; preds = %dead_block_after_break.2, %then.5
	br label %break_out
dead_block_after_goto.0:		; No predecessors!
	br label %endif.5
endif.5:		; preds = %dead_block_after_goto.0, %no_exit
	br i1 false, label %then.7, label %endif.7
then.7:		; preds = %endif.5
	br label %loopexit.1
dead_block_after_break.3:		; No predecessors!
	br label %endif.7
endif.7:		; preds = %dead_block_after_break.3, %endif.5
	switch i32 1, label %switchexit [
		 i32 4, label %label.2
		 i32 2, label %label.1
		 i32 1, label %label.0
	]
label.0:		; preds = %endif.7
	br label %switchexit
dead_block_after_break.4:		; No predecessors!
	br label %label.1
label.1:		; preds = %dead_block_after_break.4, %endif.7
	br label %switchexit
dead_block_after_break.5:		; No predecessors!
	br label %label.2
label.2:		; preds = %dead_block_after_break.5, %endif.7
	br label %switchexit
dead_block_after_break.6:		; No predecessors!
	br label %switchexit
switchexit:		; preds = %dead_block_after_break.6, %label.2, %label.1, %label.0, %endif.7
	br i1 false, label %then.8, label %endif.8
then.8:		; preds = %switchexit
	br label %loopexit.1
dead_block_after_break.7:		; No predecessors!
	br label %endif.8
endif.8:		; preds = %dead_block_after_break.7, %switchexit
	br label %loopentry.1
loopexit.1:		; preds = %then.8, %then.7, %then.6, %loopentry.1
	br i1 false, label %then.9, label %endif.9
then.9:		; preds = %loopexit.1
	br label %endif.9
endif.9:		; preds = %then.9, %loopexit.1
	br label %endif.4
else:		; preds = %shortcirc_done.4
	br i1 false, label %then.10, label %endif.10
then.10:		; preds = %else
	br label %break_out
dead_block_after_goto.1:		; No predecessors!
	br label %endif.10
endif.10:		; preds = %dead_block_after_goto.1, %else
	br label %endif.4
endif.4:		; preds = %endif.10, %endif.9
	br i1 false, label %then.11, label %endif.11
then.11:		; preds = %endif.4
	br label %loopexit.0
dead_block_after_break.8:		; No predecessors!
	br label %endif.11
endif.11:		; preds = %dead_block_after_break.8, %endif.4
	br i1 false, label %then.12, label %endif.12
then.12:		; preds = %endif.11
	br label %loopexit.0
dead_block_after_break.9:		; No predecessors!
	br label %endif.12
endif.12:		; preds = %dead_block_after_break.9, %endif.11
	br label %loopentry.0
loopexit.0:		; preds = %then.12, %then.11, %then.3, %then.2
	br label %break_out
break_out:		; preds = %loopexit.0, %then.10, %endif.6
	%retval.3 = phi i32 [ 0, %loopexit.0 ], [ %tmp.94, %endif.6 ], [ 0, %then.10 ]		; <i32> [#uses=0]
	br i1 false, label %cond_true, label %cond_false
cond_true:		; preds = %break_out
	br label %cond_continue
cond_false:		; preds = %break_out
	br label %cond_continue
cond_continue:		; preds = %cond_false, %cond_true
	br label %return
after_ret.1:		; No predecessors!
	br label %return
return:		; preds = %after_ret.1, %cond_continue, %then.1
	ret void
}
