; RUN: opt < %s -loop-reduce | llvm-dis
; PR3399

@g_53 = external global i32		; <i32*> [#uses=1]

define i32 @foo() nounwind {
bb5.thread:
	br label %bb

bb:		; preds = %bb5, %bb5.thread
	%indvar = phi i32 [ 0, %bb5.thread ], [ %indvar.next, %bb5 ]		; <i32> [#uses=2]
	br i1 false, label %bb5, label %bb1

bb1:		; preds = %bb
	%l_2.0.reg2mem.0 = sub i32 0, %indvar		; <i32> [#uses=1]
	%0 = load volatile i32* @g_53, align 4		; <i32> [#uses=1]
	%1 = trunc i32 %l_2.0.reg2mem.0 to i16		; <i16> [#uses=1]
	%2 = trunc i32 %0 to i16		; <i16> [#uses=1]
	%3 = mul i16 %2, %1		; <i16> [#uses=1]
	%4 = icmp eq i16 %3, 0		; <i1> [#uses=1]
	br i1 %4, label %bb7, label %bb2

bb2:		; preds = %bb2, %bb1
	br label %bb2

bb5:		; preds = %bb
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br label %bb

bb7:		; preds = %bb1
	ret i32 1
}
