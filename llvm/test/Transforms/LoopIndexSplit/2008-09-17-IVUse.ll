; RUN: llvm-as < %s | opt -loop-index-split -stats -disable-output | not grep "loop-index-split"
; PR 2791
@g_40 = common global i32 0		; <i32*> [#uses=1]
@g_192 = common global i32 0		; <i32*> [#uses=2]
@"\01LC" = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]

define void @func_29() nounwind {
entry:
	%0 = load i32* @g_40, align 4		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	%g_192.promoted = load i32* @g_192		; <i32> [#uses=0]
	br i1 %1, label %entry.split.us, label %entry.split

entry.split.us:		; preds = %entry
	br label %bb.us

bb.us:		; preds = %bb5.us, %entry.split.us
	%i.0.reg2mem.0.us = phi i32 [ 0, %entry.split.us ], [ %3, %bb5.us ]		; <i32> [#uses=2]
	%2 = icmp eq i32 %i.0.reg2mem.0.us, 0		; <i1> [#uses=1]
	br i1 %2, label %bb1.us, label %bb5.us

bb5.us:		; preds = %bb1.us, %bb4.us, %bb.us
	%iftmp.0.0.us = phi i32 [ 0, %bb4.us ], [ 1, %bb.us ], [ 1, %bb1.us ]		; <i32> [#uses=1]
	%3 = add i32 %i.0.reg2mem.0.us, 1		; <i32> [#uses=3]
	%4 = icmp ult i32 %3, 10		; <i1> [#uses=1]
	br i1 %4, label %bb.us, label %bb8.us

bb4.us:		; preds = %bb1.us
	br label %bb5.us

bb1.us:		; preds = %bb.us
	br i1 true, label %bb4.us, label %bb5.us

bb8.us:		; preds = %bb5.us
	%iftmp.0.0.lcssa.us = phi i32 [ %iftmp.0.0.us, %bb5.us ]		; <i32> [#uses=1]
	%.lcssa.us = phi i32 [ %3, %bb5.us ]		; <i32> [#uses=1]
	br label %bb8.split

entry.split:		; preds = %entry
	br label %bb

bb:		; preds = %bb5, %entry.split
	%i.0.reg2mem.0 = phi i32 [ 0, %entry.split ], [ %6, %bb5 ]		; <i32> [#uses=2]
	%5 = icmp eq i32 %i.0.reg2mem.0, 0		; <i1> [#uses=1]
	br i1 %5, label %bb1, label %bb5

bb1:		; preds = %bb
	br i1 false, label %bb4, label %bb5

bb4:		; preds = %bb1
	br label %bb5

bb5:		; preds = %bb1, %bb, %bb4
	%iftmp.0.0 = phi i32 [ 0, %bb4 ], [ 1, %bb ], [ 1, %bb1 ]		; <i32> [#uses=1]
	%6 = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=3]
	%7 = icmp ult i32 %6, 10		; <i1> [#uses=1]
	br i1 %7, label %bb, label %bb8

bb8:		; preds = %bb5
	%iftmp.0.0.lcssa = phi i32 [ %iftmp.0.0, %bb5 ]		; <i32> [#uses=1]
	%.lcssa = phi i32 [ %6, %bb5 ]		; <i32> [#uses=1]
	br label %bb8.split

bb8.split:		; preds = %bb8.us, %bb8
	%iftmp.0.0.lcssa.us-lcssa = phi i32 [ %iftmp.0.0.lcssa, %bb8 ], [ %iftmp.0.0.lcssa.us, %bb8.us ]		; <i32> [#uses=1]
	%.lcssa.us-lcssa = phi i32 [ %.lcssa, %bb8 ], [ %.lcssa.us, %bb8.us ]		; <i32> [#uses=1]
	store i32 %iftmp.0.0.lcssa.us-lcssa, i32* @g_192
	%8 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @"\01LC", i32 0, i32 0), i32 %.lcssa.us-lcssa ) nounwind		; <i32> [#uses=0]
	ret void
}

declare i32 @printf(i8*, ...) nounwind

define i32 @main() nounwind {
entry:
	call void @func_29( ) nounwind
	ret i32 0
}
