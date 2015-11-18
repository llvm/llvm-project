; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s

	%struct.SString = type { i8*, i32, i32 }

declare void @abort()

define fastcc void @t(%struct.SString* %word, i8 signext  %c) {
; CHECK-NOT: pop
; CHECK: bxne
; CHECK-NOT: pop
entry:
	%tmp1 = icmp eq %struct.SString* %word, null		; <i1> [#uses=1]
	br i1 %tmp1, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	tail call void @abort( )
	unreachable

cond_false:		; preds = %entry
	ret void
}
