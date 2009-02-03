; RUN: llvm-as < %s | opt -loop-index-split -stats | not grep "loop-index-split"
; PR3029

@g_138 = common global i32 0		; <i32*> [#uses=3]
@g_188 = common global i32 0		; <i32*> [#uses=4]
@g_207 = common global i32 0		; <i32*> [#uses=3]
@"\01LC" = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]
@g_102 = common global i32 0		; <i32*> [#uses=0]

define i32 @func_119() nounwind {
entry:
	%0 = volatile load i32* @g_138, align 4		; <i32> [#uses=1]
	ret i32 %0
}

define void @func_110(i32 %p_111) nounwind {
entry:
	%0 = load i32* @g_188, align 4		; <i32> [#uses=1]
	%1 = icmp ugt i32 %0, -1572397472		; <i1> [#uses=1]
	br i1 %1, label %bb, label %bb1

bb:		; preds = %entry
	%2 = volatile load i32* @g_138, align 4		; <i32> [#uses=0]
	ret void

bb1:		; preds = %entry
	store i32 1, i32* @g_207, align 4
	ret void
}

define void @func_34() nounwind {
entry:
	store i32 0, i32* @g_188
	%g_188.promoted = load i32* @g_188		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %func_110.exit, %entry
	%g_188.tmp.0 = phi i32 [ %g_188.promoted, %entry ], [ %2, %func_110.exit ]		; <i32> [#uses=2]
	%0 = icmp ugt i32 %g_188.tmp.0, -1572397472		; <i1> [#uses=1]
	br i1 %0, label %bb.i, label %bb1.i

bb.i:		; preds = %bb
	%1 = volatile load i32* @g_138, align 4		; <i32> [#uses=0]
	br label %func_110.exit

bb1.i:		; preds = %bb
	store i32 1, i32* @g_207, align 4
	br label %func_110.exit

func_110.exit:		; preds = %bb.i, %bb1.i
	%2 = add i32 %g_188.tmp.0, 1		; <i32> [#uses=3]
	%3 = icmp sgt i32 %2, 1		; <i1> [#uses=1]
	br i1 %3, label %return, label %bb

return:		; preds = %func_110.exit
	%.lcssa = phi i32 [ %2, %func_110.exit ]		; <i32> [#uses=1]
	store i32 %.lcssa, i32* @g_188
	ret void
}

define i32 @main() nounwind {
entry:
	call void @func_34() nounwind
	%0 = load i32* @g_207, align 4		; <i32> [#uses=1]
	%1 = call i32 (i8*, ...)* @printf(i8* getelementptr ([4 x i8]* @"\01LC", i32 0, i32 0), i32 %0) nounwind		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @printf(i8*, ...) nounwind
