; RUN: opt < %s -instcombine
; PR2670

@g_127 = external global i32		; <i32*> [#uses=1]

define i32 @func_56(i32 %p_58, i32 %p_59, i32 %p_61, i16 signext %p_62) nounwind {
entry:
	%call = call i32 (...) @rshift_s_s( i32 %p_61, i32 1 )		; <i32> [#uses=1]
	%conv = sext i32 %call to i64		; <i64> [#uses=1]
	%or = or i64 -1734012817166602727, %conv		; <i64> [#uses=1]
	%rem = srem i64 %or, 1		; <i64> [#uses=1]
	%cmp = icmp eq i64 %rem, 1		; <i1> [#uses=1]
	%cmp.ext = zext i1 %cmp to i32		; <i32> [#uses=1]
	store i32 %cmp.ext, i32* @g_127
	ret i32 undef
}

declare i32 @rshift_s_s(...)
