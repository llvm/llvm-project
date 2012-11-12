; RUN: opt < %s -mem2reg -disable-output
; PR670

define void @printk(i32, ...) {
entry:
	%flags = alloca i32		; <i32*> [#uses=2]
	br i1 false, label %then.0, label %endif.0
then.0:		; preds = %entry
	br label %endif.0
endif.0:		; preds = %then.0, %entry
	store i32 0, i32* %flags
	br label %loopentry
loopentry:		; preds = %endif.3, %endif.0
	br i1 false, label %no_exit, label %loopexit
no_exit:		; preds = %loopentry
	br i1 false, label %then.1, label %endif.1
then.1:		; preds = %no_exit
	br i1 false, label %shortcirc_done.0, label %shortcirc_next.0
shortcirc_next.0:		; preds = %then.1
	br label %shortcirc_done.0
shortcirc_done.0:		; preds = %shortcirc_next.0, %then.1
	br i1 false, label %shortcirc_done.1, label %shortcirc_next.1
shortcirc_next.1:		; preds = %shortcirc_done.0
	br label %shortcirc_done.1
shortcirc_done.1:		; preds = %shortcirc_next.1, %shortcirc_done.0
	br i1 false, label %shortcirc_done.2, label %shortcirc_next.2
shortcirc_next.2:		; preds = %shortcirc_done.1
	br label %shortcirc_done.2
shortcirc_done.2:		; preds = %shortcirc_next.2, %shortcirc_done.1
	br i1 false, label %then.2, label %endif.2
then.2:		; preds = %shortcirc_done.2
	br label %endif.2
endif.2:		; preds = %then.2, %shortcirc_done.2
	br label %endif.1
endif.1:		; preds = %endif.2, %no_exit
	br i1 false, label %then.3, label %endif.3
then.3:		; preds = %endif.1
	br label %endif.3
endif.3:		; preds = %then.3, %endif.1
	br label %loopentry
loopexit:		; preds = %loopentry
	br label %endif.4
then.4:		; No predecessors!
	%tmp.61 = load i32* %flags		; <i32> [#uses=0]
	br label %out
dead_block_after_goto:		; No predecessors!
	br label %endif.4
endif.4:		; preds = %dead_block_after_goto, %loopexit
	br i1 false, label %then.5, label %else
then.5:		; preds = %endif.4
	br label %endif.5
else:		; preds = %endif.4
	br label %endif.5
endif.5:		; preds = %else, %then.5
	br label %out
out:		; preds = %endif.5, %then.4
	br label %return
after_ret:		; No predecessors!
	br label %return
return:		; preds = %after_ret, %out
	ret void
}
