; RUN: llvm-as < %s | llc
; PR3117
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@g_118 = external global i8		; <i8*> [#uses=1]
@g_7 = external global i32		; <i32*> [#uses=1]

define i32 @func_73(i32 %p_74) nounwind {
entry:
	%0 = load i32* @g_7, align 4		; <i32> [#uses=1]
	%1 = or i8 0, 118		; <i8> [#uses=1]
	%2 = zext i8 %1 to i64		; <i64> [#uses=1]
	%3 = icmp ne i32 %0, 0		; <i1> [#uses=1]
	%4 = zext i1 %3 to i64		; <i64> [#uses=1]
	%5 = or i64 %4, -758998846		; <i64> [#uses=3]
	%6 = icmp sle i64 %2, %5		; <i1> [#uses=1]
	%7 = zext i1 %6 to i8		; <i8> [#uses=1]
	%8 = or i8 %7, 118		; <i8> [#uses=1]
	%9 = zext i8 %8 to i64		; <i64> [#uses=1]
	%10 = icmp sle i64 %9, 0		; <i1> [#uses=1]
	%11 = zext i1 %10 to i8		; <i8> [#uses=1]
	%12 = or i8 %11, 118		; <i8> [#uses=1]
	%13 = zext i8 %12 to i64		; <i64> [#uses=1]
	%14 = icmp sle i64 %13, %5		; <i1> [#uses=1]
	%15 = zext i1 %14 to i8		; <i8> [#uses=1]
	%16 = or i8 %15, 118		; <i8> [#uses=1]
	%17 = zext i8 %16 to i64		; <i64> [#uses=1]
	%18 = icmp sle i64 %17, 0		; <i1> [#uses=1]
	%19 = zext i1 %18 to i8		; <i8> [#uses=1]
	%20 = or i8 %19, 118		; <i8> [#uses=1]
	%21 = zext i8 %20 to i64		; <i64> [#uses=1]
	%22 = icmp sle i64 %21, %5		; <i1> [#uses=1]
	%23 = zext i1 %22 to i8		; <i8> [#uses=1]
	%24 = or i8 %23, 118		; <i8> [#uses=1]
	store i8 %24, i8* @g_118, align 1
	ret i32 undef
}
