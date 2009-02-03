; RUN: llvm-as < %s | opt -loop-index-split -stats -disable-output | not grep "1 loop-index-split"
; PR 2487
@g_6 = external global i32		; <i32*> [#uses=1]

define void @func_1() nounwind  {
entry:
	br label %bb

bb:		; preds = %bb4, %entry
	%l_3.0 = phi i8 [ 0, %entry ], [ %tmp6, %bb4 ]		; <i8> [#uses=2]
	%tmp1 = icmp eq i8 %l_3.0, 0		; <i1> [#uses=1]
	br i1 %tmp1, label %bb3, label %bb4

bb3:		; preds = %bb
	store i32 1, i32* @g_6, align 4
	br label %bb4

bb4:		; preds = %bb3, %bb
	%tmp6 = add i8 %l_3.0, 1		; <i8> [#uses=2]
	%tmp9 = icmp sgt i8 %tmp6, -1		; <i1> [#uses=1]
	br i1 %tmp9, label %bb, label %return

return:		; preds = %bb4
	ret void
}
