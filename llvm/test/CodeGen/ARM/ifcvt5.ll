; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=cortex-a8 | FileCheck %s -check-prefix=A8
; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=swift     | FileCheck %s -check-prefix=SWIFT
; rdar://8402126

@x = external global i32*		; <i32**> [#uses=1]

define void @foo(i32 %a) "no-frame-pointer-elim"="true" {
entry:
	%tmp = load i32*, i32** @x		; <i32*> [#uses=1]
	store i32 %a, i32* %tmp
	ret void
}

define i32 @t1(i32 %a, i32 %b) "no-frame-pointer-elim"="true" {
; A8-LABEL: t1:
; A8: bxlt lr

; SWIFT-LABEL: t1:
; SWIFT: bxlt lr
; SWIFT: pop {r7, pc}
entry:
	%tmp1 = icmp sgt i32 %a, 10		; <i1> [#uses=1]
	br i1 %tmp1, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	tail call void @foo( i32 %b )
	ret i32 0

UnifiedReturnBlock:		; preds = %entry
	ret i32 1
}
