; RUN: opt < %s -passes=gvn -enable-pre -S | FileCheck %s
; RUN: opt < %s -passes="gvn<pre>" -enable-pre=false -S | FileCheck %s

@H = common global i32 0		; <i32*> [#uses=2]
@G = common global i32 0		; <i32*> [#uses=1]

define i32 @test() nounwind {
entry:
	%0 = load i32, i32* @H, align 4		; <i32> [#uses=2]
	%1 = call i32 (...) @foo() nounwind		; <i32> [#uses=1]
	%2 = icmp ne i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %bb, label %bb1

bb:		; preds = %entry
	%3 = add i32 %0, 42		; <i32> [#uses=1]
; CHECK: %.pre = add i32 %0, 42
	store i32 %3, i32* @G, align 4
	br label %bb1

bb1:		; preds = %bb, %entry
	%4 = add i32 %0, 42		; <i32> [#uses=1]
	store i32 %4, i32* @H, align 4
	br label %return

; CHECK: %.pre-phi = phi i32 [ %.pre, %entry.bb1_crit_edge ], [ %3, %bb ]
; CHECK-NEXT: store i32 %.pre-phi, i32* @H, align 4
; CHECK-NEXT: ret i32 0

return:		; preds = %bb1
	ret i32 0
}

declare i32 @foo(...)
