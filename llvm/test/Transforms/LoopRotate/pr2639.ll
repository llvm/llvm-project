; RUN: opt < %s -passes=loop-deletion,loop-rotate -verify-dom-info -verify-loop-info -disable-output
; PR 2639

	%struct.HexxagonMove = type { i8, i8, i32 }

define void @_ZN16HexxagonMoveList7addMoveER12HexxagonMove() {
entry:
	br i1 false, label %bb9.preheader, label %bb11

bb9.preheader:		; preds = %entry
	br label %bb9

bb1:		; preds = %bb9
	br i1 false, label %bb3, label %bb8

bb3:		; preds = %bb1
	br label %bb5

bb4:		; preds = %bb5
	br label %bb5

bb5:		; preds = %bb4, %bb3
	%exitcond = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %exitcond, label %bb7, label %bb4

bb7:		; preds = %bb5
	store ptr null, ptr null, align 4
	br label %bb8

bb8:		; preds = %bb7, %bb1
	br label %bb9

bb9:		; preds = %bb8, %bb9.preheader
	br i1 false, label %bb11, label %bb1

bb11:		; preds = %bb9, %entry
	ret void
}
