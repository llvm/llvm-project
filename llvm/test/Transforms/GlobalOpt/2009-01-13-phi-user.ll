; RUN: opt < %s -globalopt -S | grep "phi.*@head"
; PR3321
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.node = type { %struct.node*, i32 }
@head = internal global %struct.node* null		; <%struct.node**> [#uses=2]
@node = internal global %struct.node { %struct.node* null, i32 42 }, align 16		; <%struct.node*> [#uses=1]

define i32 @f() nounwind {
entry:
	store %struct.node* @node, %struct.node** @head, align 8
	br label %bb1

bb:		; preds = %bb1
	%0 = getelementptr %struct.node* %t.0, i64 0, i32 1		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = getelementptr %struct.node* %t.0, i64 0, i32 0		; <%struct.node**> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%value.0 = phi i32 [ undef, %entry ], [ %1, %bb ]		; <i32> [#uses=1]
	%t.0.in = phi %struct.node** [ @head, %entry ], [ %2, %bb ]		; <%struct.node**> [#uses=1]
	%t.0 = load %struct.node** %t.0.in		; <%struct.node*> [#uses=3]
	%3 = icmp eq %struct.node* %t.0, null		; <i1> [#uses=1]
	br i1 %3, label %bb2, label %bb

bb2:		; preds = %bb1
	ret i32 %value.0
}

define i32 @main() nounwind {
entry:
	%0 = call i32 @f() nounwind		; <i32> [#uses=1]
	ret i32 %0
}
