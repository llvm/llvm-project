; RUN: llc < %s -march=x86-64 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9.6"
@prev_length = internal global i32 0		; <i32*> [#uses=1]
@window = internal global [65536 x i8] zeroinitializer, align 32		; <[65536 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (i32)* @longest_match to i8*)]		; <[1 x i8*]*> [#uses=0]

define fastcc i32 @longest_match(i32 %cur_match) nounwind {
; CHECK: longest_match:
; CHECK-NOT: ret
; CHECK: cmpb $0, (%r{{.*}},%r{{.*}})
; CHECK: ret

entry:
	%0 = load i32* @prev_length, align 4		; <i32> [#uses=3]
	%1 = zext i32 %cur_match to i64		; <i64> [#uses=1]
	%2 = sext i32 %0 to i64		; <i64> [#uses=1]
	%.sum3 = add i64 %1, %2		; <i64> [#uses=1]
	%3 = getelementptr [65536 x i8]* @window, i64 0, i64 %.sum3		; <i8*> [#uses=1]
	%4 = load i8* %3, align 1		; <i8> [#uses=1]
	%5 = icmp eq i8 %4, 0		; <i1> [#uses=1]
	br i1 %5, label %bb5, label %bb23

bb5:		; preds = %entry
	ret i32 %0

bb23:		; preds = %entry
	ret i32 %0
}
