; RUN: llvm-as < %s | opt -loop-index-split -stats -disable-output |& grep "1 loop-index-split" 
; PR 2869

@w = external global [2 x [2 x i32]]		; <[2 x [2 x i32]]*> [#uses=5]

declare i32 @f() nounwind

define i32 @main() noreturn nounwind {
entry:
	br label %bb1.i.outer

bb1.i.outer:		; preds = %bb5.i, %entry
	%i.0.reg2mem.0.ph.i.ph = phi i32 [ 0, %entry ], [ %indvar.next1, %bb5.i ]		; <i32> [#uses=3]
	br label %bb1.i

bb1.i:		; preds = %bb3.i, %bb1.i.outer
	%j.0.reg2mem.0.i = phi i32 [ 0, %bb1.i.outer ], [ %indvar.next, %bb3.i ]		; <i32> [#uses=3]
	%0 = icmp eq i32 %i.0.reg2mem.0.ph.i.ph, %j.0.reg2mem.0.i		; <i1> [#uses=1]
	br i1 %0, label %bb2.i, label %bb3.i

bb2.i:		; preds = %bb1.i
	%1 = getelementptr [2 x [2 x i32]]* @w, i32 0, i32 %i.0.reg2mem.0.ph.i.ph, i32 %j.0.reg2mem.0.i		; <i32*> [#uses=1]
	store i32 1, i32* %1, align 4
	br label %bb3.i

bb3.i:		; preds = %bb2.i, %bb1.i
	%indvar.next = add i32 %j.0.reg2mem.0.i, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 2		; <i1> [#uses=1]
	br i1 %exitcond, label %bb5.i, label %bb1.i

bb5.i:		; preds = %bb3.i
	%indvar.next1 = add i32 %i.0.reg2mem.0.ph.i.ph, 1		; <i32> [#uses=2]
	%exitcond2 = icmp eq i32 %indvar.next1, 2		; <i1> [#uses=1]
	br i1 %exitcond2, label %f.exit, label %bb1.i.outer

f.exit:		; preds = %bb5.i
	%2 = load i32* getelementptr ([2 x [2 x i32]]* @w, i32 0, i32 0, i32 0), align 4		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 1		; <i1> [#uses=1]
	br i1 %3, label %bb, label %bb3

bb:		; preds = %f.exit
	%4 = load i32* getelementptr ([2 x [2 x i32]]* @w, i32 0, i32 1, i32 1), align 4		; <i32> [#uses=1]
	%5 = icmp eq i32 %4, 1		; <i1> [#uses=1]
	br i1 %5, label %bb1, label %bb3

bb1:		; preds = %bb
	%6 = load i32* getelementptr ([2 x [2 x i32]]* @w, i32 0, i32 1, i32 0), align 4		; <i32> [#uses=1]
	%7 = icmp eq i32 %6, 0		; <i1> [#uses=1]
	br i1 %7, label %bb2, label %bb3

bb2:		; preds = %bb1
	%8 = load i32* getelementptr ([2 x [2 x i32]]* @w, i32 0, i32 0, i32 1), align 4		; <i32> [#uses=1]
	%9 = icmp eq i32 %8, 0		; <i1> [#uses=1]
	br i1 %9, label %bb4, label %bb3

bb3:		; preds = %bb2, %bb1, %bb, %f.exit
	tail call void @abort() noreturn nounwind
	unreachable

bb4:		; preds = %bb2
	ret i32 0
}

declare void @abort() noreturn nounwind

declare void @exit(i32) noreturn nounwind
