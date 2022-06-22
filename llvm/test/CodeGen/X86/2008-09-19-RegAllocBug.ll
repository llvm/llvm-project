; RUN: llc < %s -mtriple=i386-apple-darwin
; PR2808

@g_3 = external global i32		; <ptr> [#uses=1]

define i32 @func_4() nounwind {
entry:
	%0 = load i32, ptr @g_3, align 4		; <i32> [#uses=2]
	%1 = trunc i32 %0 to i8		; <i8> [#uses=1]
	%2 = sub i8 1, %1		; <i8> [#uses=1]
	%3 = sext i8 %2 to i32		; <i32> [#uses=1]
	%.0 = ashr i32 %3, select (i1 icmp ne (i8 zext (i1 icmp ugt (i32 ptrtoint (ptr @func_4 to i32), i32 3) to i8), i8 0), i32 0, i32 ptrtoint (ptr @func_4 to i32))		; <i32> [#uses=1]
	%4 = urem i32 %0, %.0		; <i32> [#uses=1]
	%5 = icmp eq i32 %4, 0		; <i1> [#uses=1]
	br i1 %5, label %return, label %bb4

bb4:		; preds = %entry
	ret i32 undef

return:		; preds = %entry
	ret i32 undef
}
