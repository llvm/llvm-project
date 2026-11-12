; This testcase fails because preheader insertion is not updating exit node
; information for loops.

; RUN: opt < %s -passes=licm

define i32 @main(i32 %argc, ptr %argv) {
bb0:
	br i1 false, label %bb7, label %bb5
bb5:		; preds = %bb5, %bb0
	br i1 false, label %bb5, label %bb7
bb7:		; preds = %bb7, %bb5, %bb0
	br i1 false, label %bb7, label %bb10
bb10:		; preds = %bb7
	ret i32 0
}

