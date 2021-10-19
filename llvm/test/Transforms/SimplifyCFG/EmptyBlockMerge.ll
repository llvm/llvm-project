; Basic block #2 should not be merged into BB #3!
;
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

declare void @foo()

define void @cprop_test12(i32* %data) {
bb0:
	%reg108 = load i32, i32* %data		; <i32> [#uses=2]
	%cond218 = icmp ne i32 %reg108, 5		; <i1> [#uses=1]
	br i1 %cond218, label %bb3, label %bb2
bb2:		; preds = %bb0
	call void @foo( )
; CHECK: br label %bb3
	br label %bb3
bb3:		; preds = %bb2, %bb0
	%reg117 = phi i32 [ 110, %bb2 ], [ %reg108, %bb0 ]		; <i32> [#uses=1]
	store i32 %reg117, i32* %data
	ret void
}

