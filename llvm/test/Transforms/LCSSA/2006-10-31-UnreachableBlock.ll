; RUN: opt < %s -lcssa -disable-output
; RUN: opt < %s -passes=lcssa -disable-output
; PR977
; END.

define void @process_backlog() {
entry:
	br label %loopentry.preheader
loopentry.preheader:		; preds = %dead_block_after_break, %entry
	%work.0.ph = phi i32 [ %inc, %dead_block_after_break ], [ 0, %entry ]		; <i32> [#uses=0]
	br label %loopentry
loopentry:		; preds = %endif.1, %loopentry.preheader
	br i1 false, label %then.i, label %loopentry.__skb_dequeue67.exit_crit_edge
loopentry.__skb_dequeue67.exit_crit_edge:		; preds = %loopentry
	br label %__skb_dequeue67.exit
then.i:		; preds = %loopentry
	br label %__skb_dequeue67.exit
__skb_dequeue67.exit:		; preds = %then.i, %loopentry.__skb_dequeue67.exit_crit_edge
	br i1 false, label %then.0, label %__skb_dequeue67.exit.endif.0_crit_edge
__skb_dequeue67.exit.endif.0_crit_edge:		; preds = %__skb_dequeue67.exit
	br label %endif.0
then.0:		; preds = %__skb_dequeue67.exit
	br label %job_done
dead_block_after_goto:		; No predecessors!
	unreachable
endif.0:		; preds = %__skb_dequeue67.exit.endif.0_crit_edge
	br i1 false, label %then.0.i, label %endif.0.endif.0.i_crit_edge
endif.0.endif.0.i_crit_edge:		; preds = %endif.0
	br label %endif.0.i
then.0.i:		; preds = %endif.0
	br label %endif.0.i
endif.0.i:		; preds = %then.0.i, %endif.0.endif.0.i_crit_edge
	br i1 false, label %then.i.i, label %endif.0.i.skb_bond.exit.i_crit_edge
endif.0.i.skb_bond.exit.i_crit_edge:		; preds = %endif.0.i
	br label %skb_bond.exit.i
then.i.i:		; preds = %endif.0.i
	br label %skb_bond.exit.i
skb_bond.exit.i:		; preds = %then.i.i, %endif.0.i.skb_bond.exit.i_crit_edge
	br label %loopentry.0.i
loopentry.0.i:		; preds = %loopentry.0.i.backedge, %skb_bond.exit.i
	br i1 false, label %loopentry.0.i.no_exit.0.i_crit_edge, label %loopentry.0.i.loopexit.0.i_crit_edge
loopentry.0.i.loopexit.0.i_crit_edge:		; preds = %loopentry.0.i
	br label %loopexit.0.i
loopentry.0.i.no_exit.0.i_crit_edge:		; preds = %loopentry.0.i
	br label %no_exit.0.i
no_exit.0.i:		; preds = %then.3.i.no_exit.0.i_crit_edge, %loopentry.0.i.no_exit.0.i_crit_edge
	br i1 false, label %no_exit.0.i.shortcirc_done.0.i_crit_edge, label %shortcirc_next.0.i
no_exit.0.i.shortcirc_done.0.i_crit_edge:		; preds = %no_exit.0.i
	br label %shortcirc_done.0.i
shortcirc_next.0.i:		; preds = %no_exit.0.i
	br label %shortcirc_done.0.i
shortcirc_done.0.i:		; preds = %shortcirc_next.0.i, %no_exit.0.i.shortcirc_done.0.i_crit_edge
	br i1 false, label %then.1.i, label %endif.1.i
then.1.i:		; preds = %shortcirc_done.0.i
	br i1 false, label %then.2.i, label %then.1.i.endif.2.i_crit_edge
then.1.i.endif.2.i_crit_edge:		; preds = %then.1.i
	br label %endif.2.i
then.2.i:		; preds = %then.1.i
	br i1 false, label %then.3.i, label %else.0.i
then.3.i:		; preds = %then.2.i
	br i1 false, label %then.3.i.no_exit.0.i_crit_edge, label %then.3.i.loopexit.0.i_crit_edge
then.3.i.loopexit.0.i_crit_edge:		; preds = %then.3.i
	br label %loopexit.0.i
then.3.i.no_exit.0.i_crit_edge:		; preds = %then.3.i
	br label %no_exit.0.i
else.0.i:		; preds = %then.2.i
	br label %endif.2.i
endif.3.i:		; No predecessors!
	unreachable
endif.2.i:		; preds = %else.0.i, %then.1.i.endif.2.i_crit_edge
	br label %loopentry.0.i.backedge
endif.1.i:		; preds = %shortcirc_done.0.i
	br label %loopentry.0.i.backedge
loopentry.0.i.backedge:		; preds = %endif.1.i, %endif.2.i
	br label %loopentry.0.i
loopexit.0.i:		; preds = %then.3.i.loopexit.0.i_crit_edge, %loopentry.0.i.loopexit.0.i_crit_edge
	br label %loopentry.1.i
loopentry.1.i:		; preds = %loopentry.1.i.backedge, %loopexit.0.i
	br i1 false, label %loopentry.1.i.no_exit.1.i_crit_edge, label %loopentry.1.i.loopexit.1.i_crit_edge
loopentry.1.i.loopexit.1.i_crit_edge:		; preds = %loopentry.1.i
	br label %loopexit.1.i
loopentry.1.i.no_exit.1.i_crit_edge:		; preds = %loopentry.1.i
	br label %no_exit.1.i
no_exit.1.i:		; preds = %then.6.i.no_exit.1.i_crit_edge, %loopentry.1.i.no_exit.1.i_crit_edge
	br i1 false, label %shortcirc_next.1.i, label %no_exit.1.i.shortcirc_done.1.i_crit_edge
no_exit.1.i.shortcirc_done.1.i_crit_edge:		; preds = %no_exit.1.i
	br label %shortcirc_done.1.i
shortcirc_next.1.i:		; preds = %no_exit.1.i
	br i1 false, label %shortcirc_next.1.i.shortcirc_done.2.i_crit_edge, label %shortcirc_next.2.i
shortcirc_next.1.i.shortcirc_done.2.i_crit_edge:		; preds = %shortcirc_next.1.i
	br label %shortcirc_done.2.i
shortcirc_next.2.i:		; preds = %shortcirc_next.1.i
	br label %shortcirc_done.2.i
shortcirc_done.2.i:		; preds = %shortcirc_next.2.i, %shortcirc_next.1.i.shortcirc_done.2.i_crit_edge
	br label %shortcirc_done.1.i
shortcirc_done.1.i:		; preds = %shortcirc_done.2.i, %no_exit.1.i.shortcirc_done.1.i_crit_edge
	br i1 false, label %then.4.i, label %endif.4.i
then.4.i:		; preds = %shortcirc_done.1.i
	br i1 false, label %then.5.i, label %then.4.i.endif.5.i_crit_edge
then.4.i.endif.5.i_crit_edge:		; preds = %then.4.i
	br label %endif.5.i
then.5.i:		; preds = %then.4.i
	br i1 false, label %then.6.i, label %else.1.i
then.6.i:		; preds = %then.5.i
	br i1 false, label %then.6.i.no_exit.1.i_crit_edge, label %then.6.i.loopexit.1.i_crit_edge
then.6.i.loopexit.1.i_crit_edge:		; preds = %then.6.i
	br label %loopexit.1.i
then.6.i.no_exit.1.i_crit_edge:		; preds = %then.6.i
	br label %no_exit.1.i
else.1.i:		; preds = %then.5.i
	br label %endif.5.i
endif.6.i:		; No predecessors!
	unreachable
endif.5.i:		; preds = %else.1.i, %then.4.i.endif.5.i_crit_edge
	br label %loopentry.1.i.backedge
endif.4.i:		; preds = %shortcirc_done.1.i
	br label %loopentry.1.i.backedge
loopentry.1.i.backedge:		; preds = %endif.4.i, %endif.5.i
	br label %loopentry.1.i
loopexit.1.i:		; preds = %then.6.i.loopexit.1.i_crit_edge, %loopentry.1.i.loopexit.1.i_crit_edge
	br i1 false, label %then.7.i, label %else.2.i
then.7.i:		; preds = %loopexit.1.i
	br i1 false, label %then.8.i, label %else.3.i
then.8.i:		; preds = %then.7.i
	br label %netif_receive_skb.exit
else.3.i:		; preds = %then.7.i
	br label %netif_receive_skb.exit
endif.8.i:		; No predecessors!
	unreachable
else.2.i:		; preds = %loopexit.1.i
	br i1 false, label %else.2.i.shortcirc_done.i.i_crit_edge, label %shortcirc_next.i.i
else.2.i.shortcirc_done.i.i_crit_edge:		; preds = %else.2.i
	br label %shortcirc_done.i.i
shortcirc_next.i.i:		; preds = %else.2.i
	br label %shortcirc_done.i.i
shortcirc_done.i.i:		; preds = %shortcirc_next.i.i, %else.2.i.shortcirc_done.i.i_crit_edge
	br i1 false, label %then.i1.i, label %shortcirc_done.i.i.kfree_skb65.exit.i_crit_edge
shortcirc_done.i.i.kfree_skb65.exit.i_crit_edge:		; preds = %shortcirc_done.i.i
	br label %kfree_skb65.exit.i
then.i1.i:		; preds = %shortcirc_done.i.i
	br label %kfree_skb65.exit.i
kfree_skb65.exit.i:		; preds = %then.i1.i, %shortcirc_done.i.i.kfree_skb65.exit.i_crit_edge
	br label %netif_receive_skb.exit
netif_receive_skb.exit:		; preds = %kfree_skb65.exit.i, %else.3.i, %then.8.i
	br i1 false, label %then.i1, label %netif_receive_skb.exit.dev_put69.exit_crit_edge
netif_receive_skb.exit.dev_put69.exit_crit_edge:		; preds = %netif_receive_skb.exit
	br label %dev_put69.exit
then.i1:		; preds = %netif_receive_skb.exit
	br label %dev_put69.exit
dev_put69.exit:		; preds = %then.i1, %netif_receive_skb.exit.dev_put69.exit_crit_edge
	%inc = add i32 0, 1		; <i32> [#uses=1]
	br i1 false, label %dev_put69.exit.shortcirc_done_crit_edge, label %shortcirc_next
dev_put69.exit.shortcirc_done_crit_edge:		; preds = %dev_put69.exit
	br label %shortcirc_done
shortcirc_next:		; preds = %dev_put69.exit
	br label %shortcirc_done
shortcirc_done:		; preds = %shortcirc_next, %dev_put69.exit.shortcirc_done_crit_edge
	br i1 false, label %then.1, label %endif.1
then.1:		; preds = %shortcirc_done
	ret void
dead_block_after_break:		; No predecessors!
	br label %loopentry.preheader
endif.1:		; preds = %shortcirc_done
	br label %loopentry
loopexit:		; No predecessors!
	unreachable
after_ret.0:		; No predecessors!
	br label %job_done
job_done:		; preds = %after_ret.0, %then.0
	br label %loopentry.i
loopentry.i:		; preds = %no_exit.i, %job_done
	br i1 false, label %no_exit.i, label %clear_bit62.exit
no_exit.i:		; preds = %loopentry.i
	br label %loopentry.i
clear_bit62.exit:		; preds = %loopentry.i
	br i1 false, label %then.2, label %endif.2
then.2:		; preds = %clear_bit62.exit
	ret void
endif.2:		; preds = %clear_bit62.exit
	ret void
after_ret.1:		; No predecessors!
	ret void
return:		; No predecessors!
	unreachable
}
