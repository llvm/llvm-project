; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK: phi{{.*}}@head
; PR3321
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.node = type { ptr, i32 }
@head = internal global ptr null		; <ptr> [#uses=2]
@node = internal global %struct.node { ptr null, i32 42 }, align 16		; <ptr> [#uses=1]

define i32 @f() nounwind {
entry:
	store ptr @node, ptr @head, align 8
	br label %bb1

bb:		; preds = %bb1
	%0 = getelementptr %struct.node, ptr %t.0, i64 0, i32 1		; <ptr> [#uses=1]
	%1 = load i32, ptr %0, align 4		; <i32> [#uses=1]
	%2 = getelementptr %struct.node, ptr %t.0, i64 0, i32 0		; <ptr> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%value.0 = phi i32 [ undef, %entry ], [ %1, %bb ]		; <i32> [#uses=1]
	%t.0.in = phi ptr [ @head, %entry ], [ %2, %bb ]		; <ptr> [#uses=1]
	%t.0 = load ptr, ptr %t.0.in		; <ptr> [#uses=3]
	%3 = icmp eq ptr %t.0, null		; <i1> [#uses=1]
	br i1 %3, label %bb2, label %bb

bb2:		; preds = %bb1
	ret i32 %value.0
}

define i32 @main() nounwind {
entry:
	%0 = call i32 @f() nounwind		; <i32> [#uses=1]
	ret i32 %0
}
