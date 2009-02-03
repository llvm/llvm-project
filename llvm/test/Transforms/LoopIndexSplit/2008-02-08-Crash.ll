; RUN: llvm-as < %s | opt -loop-index-split -disable-output
; PR 1995

define void @add_blkdev_randomness(i32 %major) nounwind  {
entry:
	br label %bb

bb:		; preds = %bb39, %entry
	%A.0.reg2mem.0 = phi i32 [ undef, %entry ], [ %TEMP.0, %bb39 ]		; <i32> [#uses=1]
	%D.0.reg2mem.0 = phi i32 [ undef, %entry ], [ %C.0.reg2mem.0, %bb39 ]		; <i32> [#uses=3]
	%C.0.reg2mem.0 = phi i32 [ undef, %entry ], [ %tmp34, %bb39 ]		; <i32> [#uses=4]
	%TEMP.1.reg2mem.0 = phi i32 [ undef, %entry ], [ %TEMP.0, %bb39 ]		; <i32> [#uses=1]
	%i.0.reg2mem.0 = phi i32 [ 0, %entry ], [ %tmp38, %bb39 ]		; <i32> [#uses=3]
	%B.0.reg2mem.0 = phi i32 [ undef, %entry ], [ %A.0.reg2mem.0, %bb39 ]		; <i32> [#uses=5]
	%tmp1 = icmp slt i32 %i.0.reg2mem.0, 40		; <i1> [#uses=1]
	br i1 %tmp1, label %bb3, label %bb12

bb3:		; preds = %bb
	%tmp6 = xor i32 %C.0.reg2mem.0, %D.0.reg2mem.0		; <i32> [#uses=1]
	%tmp8 = and i32 %B.0.reg2mem.0, %tmp6		; <i32> [#uses=1]
	%tmp10 = xor i32 %tmp8, %D.0.reg2mem.0		; <i32> [#uses=1]
	%tmp11 = add i32 %tmp10, 1518500249		; <i32> [#uses=1]
	br label %bb39

bb12:		; preds = %bb
	%tmp14 = icmp slt i32 %i.0.reg2mem.0, 60		; <i1> [#uses=1]
	br i1 %tmp14, label %bb17, label %bb39

bb17:		; preds = %bb12
	%tmp20 = and i32 %B.0.reg2mem.0, %C.0.reg2mem.0		; <i32> [#uses=1]
	%tmp23 = xor i32 %B.0.reg2mem.0, %C.0.reg2mem.0		; <i32> [#uses=1]
	%tmp25 = and i32 %tmp23, %D.0.reg2mem.0		; <i32> [#uses=1]
	%tmp26 = add i32 %tmp20, -1894007588		; <i32> [#uses=1]
	%tmp27 = add i32 %tmp26, %tmp25		; <i32> [#uses=1]
	br label %bb39

bb39:		; preds = %bb12, %bb3, %bb17
	%TEMP.0 = phi i32 [ %tmp27, %bb17 ], [ %tmp11, %bb3 ], [ %TEMP.1.reg2mem.0, %bb12 ]		; <i32> [#uses=2]
	%tmp31 = lshr i32 %B.0.reg2mem.0, 2		; <i32> [#uses=1]
	%tmp33 = shl i32 %B.0.reg2mem.0, 30		; <i32> [#uses=1]
	%tmp34 = or i32 %tmp31, %tmp33		; <i32> [#uses=1]
	%tmp38 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%tmp41 = icmp slt i32 %tmp38, 80		; <i1> [#uses=1]
	br i1 %tmp41, label %bb, label %return

return:		; preds = %bb39
	ret void
}