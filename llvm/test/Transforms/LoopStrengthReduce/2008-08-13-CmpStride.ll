; RUN: opt < %s -loop-reduce -S | grep add | count 2
; PR 2662
@g_3 = common global i16 0		; <i16*> [#uses=2]
@"\01LC" = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]

define void @func_1() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%l_2.0.reg2mem.0 = phi i16 [ 0, %entry ], [ %t1, %bb ]		; <i16> [#uses=2]
	%t0 = shl i16 %l_2.0.reg2mem.0, 1		; <i16>:0 [#uses=1]
	store volatile i16 %t0, i16* @g_3, align 2
	%t1 = add i16 %l_2.0.reg2mem.0, -3		; <i16>:1 [#uses=2]
	%t2 = icmp slt i16 %t1, 1		; <i1>:2 [#uses=1]
	br i1 %t2, label %bb, label %return

return:		; preds = %bb
	ret void
}

define i32 @main() nounwind {
entry:
	tail call void @func_1( ) nounwind
	load volatile i16* @g_3, align 2		; <i16>:0 [#uses=1]
	zext i16 %0 to i32		; <i32>:1 [#uses=1]
	tail call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @"\01LC", i32 0, i32 0), i32 %1 ) nounwind		; <i32>:2 [#uses=0]
	ret i32 0
}

declare i32 @printf(i8*, ...) nounwind
