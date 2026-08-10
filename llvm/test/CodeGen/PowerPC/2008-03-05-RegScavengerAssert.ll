; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

declare ptr @bar(i32)

; CHECK: @foo
; CHECK: blr

define void @foo(ptr %pp) nounwind  {
entry:
	%tmp2 = tail call ptr @bar( i32 14 ) nounwind 		; <ptr> [#uses=0]
	%tmp38 = load ptr, ptr %pp, align 4		; <ptr> [#uses=2]
	br i1 false, label %bb34, label %bb25
bb25:		; preds = %entry
	tail call void %tmp38( ptr null ) nounwind 
	ret void
bb34:		; preds = %entry
	tail call void %tmp38( ) nounwind 
	ret void
}
