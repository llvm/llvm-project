; RUN: llvm-as < %s | llc
; PR2735
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
@g_385 = external global i32		; <i32*> [#uses=1]

define i32 @func_45(i64 %p_46, i32 %p_48) nounwind {
entry:
	%0 = tail call i32 (...)* @lshift_s_u(i64 %p_46, i64 0) nounwind		; <i32> [#uses=0]
	%1 = load i32* @g_385, align 4		; <i32> [#uses=1]
	%2 = shl i32 %1, 1		; <i32> [#uses=1]
	%3 = and i32 %2, 32		; <i32> [#uses=1]
	%4 = tail call i32 (...)* @func_87(i32 undef, i32 %p_48, i32 1) nounwind		; <i32> [#uses=1]
	%5 = add i32 %3, %4		; <i32> [#uses=1]
	%6 = tail call i32 (...)* @div_rhs(i32 %5) nounwind		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @lshift_s_u(...)
declare i32 @func_87(...)
declare i32 @div_rhs(...)
