; RUN: llc < %s | FileCheck %s
; this should not sink %1 into bb1, that would increase reg pressure.

; rdar://6399178

; CHECK: addl $4,
; CHECK-NOT: leal

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

define i32 @bar(ptr %P) nounwind {
entry:
	%0 = load ptr, ptr %P, align 4		; <ptr> [#uses=2]
	%1 = getelementptr i32, ptr %0, i32 1		; <ptr> [#uses=1]
	%2 = icmp ugt ptr %1, inttoptr (i64 1233 to ptr)		; <i1> [#uses=1]
	br i1 %2, label %bb1, label %bb

bb:		; preds = %entry
	store ptr inttoptr (i64 123 to ptr), ptr %P, align 4
	br label %bb1

bb1:		; preds = %entry, %bb
	%3 = getelementptr i32, ptr %1, i32 1		; <ptr> [#uses=1]
	%4 = load i32, ptr %3, align 4		; <i32> [#uses=1]
	ret i32 %4
}
