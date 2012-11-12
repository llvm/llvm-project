; RUN: opt < %s -simplifycfg -disable-output
; PR 2777
@g_103 = common global i32 0		; <i32*> [#uses=1]

define i32 @func_127(i32 %p_129) nounwind {
entry:
	load i32* @g_103, align 4		; <i32>:0 [#uses=1]
	icmp eq i32 %0, 0		; <i1>:1 [#uses=2]
	br i1 %1, label %bb6.preheader, label %entry.return_crit_edge

entry.return_crit_edge:		; preds = %entry
	br label %return

bb6.preheader:		; preds = %entry
	br i1 %1, label %bb6.preheader.split.us, label %bb6.preheader.split

bb6.preheader.split.us:		; preds = %bb6.preheader
	br label %return.loopexit.split

bb6.preheader.split:		; preds = %bb6.preheader
	br label %bb6

bb6:		; preds = %bb17.bb6_crit_edge, %bb6.preheader.split
	%indvar35 = phi i32 [ 0, %bb6.preheader.split ], [ %indvar.next36, %bb17.bb6_crit_edge ]		; <i32> [#uses=1]
	%p_129_addr.3.reg2mem.0 = phi i32 [ %p_129_addr.2, %bb17.bb6_crit_edge ], [ %p_129, %bb6.preheader.split ]		; <i32> [#uses=3]
	icmp eq i32 %p_129_addr.3.reg2mem.0, 0		; <i1>:2 [#uses=1]
	br i1 %2, label %bb6.bb17_crit_edge, label %bb8

bb6.bb17_crit_edge:		; preds = %bb6
	br label %bb17

bb8:		; preds = %bb6
	br label %bb13

bb13:		; preds = %bb8
	br label %bb17

bb17:		; preds = %bb13, %bb6.bb17_crit_edge
	%p_129_addr.2 = phi i32 [ %p_129_addr.3.reg2mem.0, %bb13 ], [ %p_129_addr.3.reg2mem.0, %bb6.bb17_crit_edge ]		; <i32> [#uses=1]
	%indvar.next36 = add i32 %indvar35, 1		; <i32> [#uses=2]
	%exitcond37 = icmp eq i32 %indvar.next36, -1		; <i1> [#uses=1]
	br i1 %exitcond37, label %return.loopexit, label %bb17.bb6_crit_edge

bb17.bb6_crit_edge:		; preds = %bb17
	br label %bb6

return.loopexit:		; preds = %bb17
	br label %return.loopexit.split

return.loopexit.split:		; preds = %return.loopexit, %bb6.preheader.split.us
	br label %return

return:		; preds = %return.loopexit.split, %entry.return_crit_edge
	ret i32 1
}

define i32 @func_135(i8 zeroext %p_137, i32 %p_138, i32 %p_140) nounwind {
entry:
	ret i32 undef
}
