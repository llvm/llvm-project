; RUN: llvm-as < %s | llc
; PR3044
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f128:128:128"
target triple = "alphaev6-unknown-linux-gnu"

define i128 @__mulvti3(i128 %u, i128 %v) nounwind {
entry:
	%0 = load i128* null, align 16		; <i128> [#uses=1]
	%1 = load i64* null, align 8		; <i64> [#uses=1]
	%2 = zext i64 %1 to i128		; <i128> [#uses=1]
	%3 = add i128 %2, %0		; <i128> [#uses=1]
	store i128 %3, i128* null, align 16
	unreachable
}
