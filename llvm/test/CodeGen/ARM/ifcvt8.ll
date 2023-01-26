; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s

	%struct.SString = type { ptr, i32, i32 }

declare void @abort()

define fastcc void @t(ptr %word, i8 signext  %c) {
; CHECK-NOT: pop
; CHECK: bxne
; CHECK-NOT: pop
entry:
	%tmp1 = icmp eq ptr %word, null		; <i1> [#uses=1]
	br i1 %tmp1, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	tail call void @abort( )
	unreachable

cond_false:		; preds = %entry
	ret void
}
