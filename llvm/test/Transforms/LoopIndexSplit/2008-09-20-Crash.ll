; RUN: llvm-as < %s | opt -loop-index-split -disable-output
; PR 2805
@g_330 = common global i32 0		; <i32*> [#uses=1]

define i32 @func_45(i32 %p_47) nounwind {
entry:
	br label %bb

bb:		; preds = %bb3, %entry
	%p_47_addr.0.reg2mem.0 = phi i32 [ 0, %entry ], [ %2, %bb3 ]		; <i32> [#uses=2]
	%0 = icmp eq i32 %p_47_addr.0.reg2mem.0, 0		; <i1> [#uses=1]
	br i1 %0, label %bb2, label %bb1

bb1:		; preds = %bb
	%1 = tail call i32 (...)* @func_70( i32 1 ) nounwind		; <i32> [#uses=0]
	br label %bb3

bb2:		; preds = %bb
	store i32 1, i32* @g_330, align 4
	br label %bb3

bb3:		; preds = %bb2, %bb1
	%2 = add i32 %p_47_addr.0.reg2mem.0, 1		; <i32> [#uses=3]
	%3 = icmp ult i32 %2, 22		; <i1> [#uses=1]
	br i1 %3, label %bb, label %bb6

bb6:		; preds = %bb3
	%.lcssa = phi i32 [ %2, %bb3 ]		; <i32> [#uses=1]
	%4 = tail call i32 (...)* @func_95( i32 %.lcssa ) nounwind		; <i32> [#uses=1]
	%5 = tail call i32 (...)* @func_56( i32 %4 ) nounwind		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @func_70(...)

declare i32 @func_95(...)

declare i32 @func_56(...)
