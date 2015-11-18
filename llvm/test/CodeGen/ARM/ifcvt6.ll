; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s

define void @foo(i32 %X, i32 %Y) {
entry:
; CHECK: cmpne
; CHECK: bxhi lr
	%tmp1 = icmp ult i32 %X, 4		; <i1> [#uses=1]
	%tmp4 = icmp eq i32 %Y, 0		; <i1> [#uses=1]
	%tmp7 = or i1 %tmp4, %tmp1		; <i1> [#uses=1]
	br i1 %tmp7, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	%tmp10 = call i32 (...) @bar( )		; <i32> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare i32 @bar(...)
