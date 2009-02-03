; RUN: llvm-as < %s | opt -loop-index-split -stats -disable-output | not grep "loop-index-split"
;PR2294
@g_2 = external global i16		; <i16*> [#uses=4]
@g_5 = external global i32		; <i32*> [#uses=1]
@.str = external constant [4 x i8]		; <[4 x i8]*> [#uses=1]

declare void @func_1() nounwind 

define i32 @main() nounwind  {
entry:
	%tmp101.i = load i16* @g_2, align 2		; <i16> [#uses=1]
	%tmp112.i = icmp sgt i16 %tmp101.i, 0		; <i1> [#uses=1]
	br i1 %tmp112.i, label %bb.preheader.i, label %func_1.exit
bb.preheader.i:		; preds = %entry
	%g_2.promoted.i = load i16* @g_2		; <i16> [#uses=1]
	br label %bb.i
bb.i:		; preds = %bb6.i, %bb.preheader.i
	%g_2.tmp.0.i = phi i16 [ %g_2.promoted.i, %bb.preheader.i ], [ %tmp8.i, %bb6.i ]		; <i16> [#uses=2]
	%tmp2.i = icmp eq i16 %g_2.tmp.0.i, 0		; <i1> [#uses=1]
	br i1 %tmp2.i, label %bb4.i, label %bb6.i
bb4.i:		; preds = %bb.i
	%tmp5.i = volatile load i32* @g_5, align 4		; <i32> [#uses=0]
	br label %bb6.i
bb6.i:		; preds = %bb4.i, %bb.i
	%tmp8.i = add i16 %g_2.tmp.0.i, 1		; <i16> [#uses=3]
	%tmp11.i = icmp sgt i16 %tmp8.i, 42		; <i1> [#uses=1]
	br i1 %tmp11.i, label %bb.i, label %return.loopexit.i
return.loopexit.i:		; preds = %bb6.i
	%tmp8.i.lcssa = phi i16 [ %tmp8.i, %bb6.i ]		; <i16> [#uses=1]
	store i16 %tmp8.i.lcssa, i16* @g_2
	br label %func_1.exit
func_1.exit:		; preds = %return.loopexit.i, %entry
	%tmp1 = load i16* @g_2, align 2		; <i16> [#uses=1]
	%tmp12 = sext i16 %tmp1 to i32		; <i32> [#uses=1]
	%tmp3 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0), i32 %tmp12 ) nounwind 		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @printf(i8*, ...) nounwind 

