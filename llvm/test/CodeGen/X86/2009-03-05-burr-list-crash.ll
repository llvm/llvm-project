; RUN: llc < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@0 = external global i32		; <ptr>:0 [#uses=1]

declare i64 @strlen(ptr nocapture) nounwind readonly

define fastcc ptr @1(ptr) nounwind {
	br i1 false, label %3, label %2

; <label>:2		; preds = %1
	ret ptr %0

; <label>:3		; preds = %1
	%4 = call i64 @strlen(ptr %0) nounwind readonly		; <i64> [#uses=1]
	%5 = trunc i64 %4 to i32		; <i32> [#uses=2]
	%6 = load i32, ptr @0, align 4		; <i32> [#uses=1]
	%7 = sub i32 %5, %6		; <i32> [#uses=2]
	%8 = sext i32 %5 to i64		; <i64> [#uses=1]
	%9 = sext i32 %7 to i64		; <i64> [#uses=1]
	%10 = sub i64 %8, %9		; <i64> [#uses=1]
	%11 = getelementptr i8, ptr %0, i64 %10		; <ptr> [#uses=1]
	%12 = icmp sgt i32 %7, 0		; <i1> [#uses=1]
	br i1 %12, label %13, label %14

; <label>:13		; preds = %13, %3
	br label %13

; <label>:14		; preds = %3
	%15 = call noalias ptr @make_temp_file(ptr %11) nounwind		; <ptr> [#uses=0]
	unreachable
}

declare noalias ptr @make_temp_file(ptr)
