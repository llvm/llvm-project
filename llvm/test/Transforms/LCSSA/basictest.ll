; RUN: opt < %s -lcssa -S | FileCheck %s
; RUN: opt < %s -passes=lcssa -S | FileCheck %s

define void @lcssa(i1 %S2) {
; CHECK-LABEL: @lcssa
entry:
	br label %loop.interior
loop.interior:		; preds = %post.if, %entry
	br i1 %S2, label %if.true, label %if.false
if.true:		; preds = %loop.interior
	%X1 = add i32 0, 0		; <i32> [#uses=1]
	br label %post.if
if.false:		; preds = %loop.interior
	%X2 = add i32 0, 1		; <i32> [#uses=1]
	br label %post.if
post.if:		; preds = %if.false, %if.true
	%X3 = phi i32 [ %X1, %if.true ], [ %X2, %if.false ]		; <i32> [#uses=1]
	br i1 %S2, label %loop.exit, label %loop.interior
loop.exit:		; preds = %post.if
; CHECK: %X3.lcssa = phi i32
; CHECK: %X4 = add i32 3, %X3.lcssa
	%X4 = add i32 3, %X3		; <i32> [#uses=0]
	ret void
}

