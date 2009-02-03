; This test shows SCCP "proving" that the loop (from bb6 to 14) loops infinitely
; this is in fact NOT the case, so the return should still be alive in the code
; after sccp and CFG simplification have been performed.
;
; RUN: llvm-as < %s | opt -sccp -simplifycfg | llvm-dis | \
; RUN:   grep ret

define void @old_main() {
bb3:
	br label %bb6
bb6:		; preds = %bb14, %bb3
	%reg403 = phi i32 [ %reg155, %bb14 ], [ 0, %bb3 ]		; <i32> [#uses=1]
	%reg155 = add i32 %reg403, 1		; <i32> [#uses=2]
	br label %bb11
bb11:		; preds = %bb11, %bb6
	%reg407 = phi i32 [ %reg408, %bb11 ], [ 0, %bb6 ]		; <i32> [#uses=2]
	%reg408 = add i32 %reg407, 1		; <i32> [#uses=1]
	%cond550 = icmp sle i32 %reg407, 1		; <i1> [#uses=1]
	br i1 %cond550, label %bb11, label %bb12
bb12:		; preds = %bb11
	br label %bb13
bb13:		; preds = %bb13, %bb12
	%reg409 = phi i32 [ %reg410, %bb13 ], [ 0, %bb12 ]		; <i32> [#uses=1]
	%reg410 = add i32 %reg409, 1		; <i32> [#uses=2]
	%cond552 = icmp sle i32 %reg410, 2		; <i1> [#uses=1]
	br i1 %cond552, label %bb13, label %bb14
bb14:		; preds = %bb13
	%cond553 = icmp sle i32 %reg155, 31		; <i1> [#uses=1]
	br i1 %cond553, label %bb6, label %bb15
bb15:		; preds = %bb14
	ret void
}

