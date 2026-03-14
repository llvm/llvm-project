; RUN: opt < %s -loop-reduce -S | grep phi | count 2
; PR 2779

; Provide legal integer types.
target datalayout = "n8:16:32:64"

@g_19 = common global i32 0		; <ptr> [#uses=3]
@"\01LC" = internal constant [4 x i8] c"%d\0A\00"		; <ptr> [#uses=1]

define i32 @func_8(i8 zeroext %p_9) nounwind {
entry:
	ret i32 1
}

define i32 @func_3(i8 signext %p_5) nounwind {
entry:
	ret i32 1
}

define void @func_1() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%indvar = phi i16 [ 0, %entry ], [ %indvar.next, %bb ]		; <i16> [#uses=2]
	%tmp = sub i16 0, %indvar		; <i16> [#uses=1]
	%tmp27 = trunc i16 %tmp to i8		; <i8> [#uses=1]
	load i32, ptr @g_19, align 4		; <i32>:0 [#uses=2]
	add i32 %0, 1		; <i32>:1 [#uses=1]
	store i32 %1, ptr @g_19, align 4
	trunc i32 %0 to i8		; <i8>:2 [#uses=1]
	tail call i32 @func_8( i8 zeroext %2 ) nounwind		; <i32>:3 [#uses=0]
	shl i8 %tmp27, 2		; <i8>:4 [#uses=1]
	add i8 %4, -112		; <i8>:5 [#uses=1]
	tail call i32 @func_3( i8 signext %5 ) nounwind		; <i32>:6 [#uses=0]
	%indvar.next = add i16 %indvar, 1		; <i16> [#uses=2]
	%exitcond = icmp eq i16 %indvar.next, -28		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb
	ret void
}

define i32 @main() nounwind {
entry:
	tail call void @func_1( ) nounwind
	load i32, ptr @g_19, align 4		; <i32>:0 [#uses=1]
	tail call i32 (ptr, ...) @printf( ptr @"\01LC", i32 %0 ) nounwind		; <i32>:1 [#uses=0]
	ret i32 0
}

declare i32 @printf(ptr, ...) nounwind
