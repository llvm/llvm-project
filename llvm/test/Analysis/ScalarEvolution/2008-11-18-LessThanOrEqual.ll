; RUN: opt < %s -analyze -scalar-evolution 2>&1 | \
; RUN: grep "Loop %bb: backedge-taken count is (7 + (-1 \* %argc))"

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	%0 = icmp ugt i32 %argc, 7		; <i1> [#uses=1]
	br i1 %0, label %bb2, label %bb.nph

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb.nph, %bb1
	%indvar = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb1 ]		; <i32> [#uses=2]
	%argc_addr.04 = add i32 %indvar, %argc		; <i32> [#uses=1]
	tail call void (...)* @Test() nounwind
	%1 = add i32 %argc_addr.04, 1		; <i32> [#uses=1]
	br label %bb1

bb1:		; preds = %bb
	%phitmp = icmp ugt i32 %1, 7		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %phitmp, label %bb1.bb2_crit_edge, label %bb

bb1.bb2_crit_edge:		; preds = %bb1
	br label %bb2

bb2:		; preds = %bb1.bb2_crit_edge, %entry
	ret i32 0
}

declare void @Test(...)
