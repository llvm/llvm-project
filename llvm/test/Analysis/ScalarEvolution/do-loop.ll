; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s
; PR1614

; CHECK: smax

define i32 @f(i32 %x, i32 %y) {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%x_addr.0 = add i32 %indvar, %x		; <i32> [#uses=1]
	%tmp2 = add i32 %x_addr.0, 1		; <i32> [#uses=2]
	%tmp5 = icmp slt i32 %tmp2, %y		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp5, label %bb, label %bb7

bb7:		; preds = %bb
	ret i32 %tmp2
}
