; RUN: opt < %s -passes=indvars -S | FileCheck %s

; Indvars should insert a 64-bit induction variable to eliminate the
; sext for the addressing, however it shouldn't eliminate the sext
; on the other phi, since that value undergoes signed wrapping.

; Provide legal integer types.
target datalayout = "n8:16:32:64"


define void @foo(i32* nocapture %d, i32 %n) nounwind {
entry:
	%0 = icmp sgt i32 %n, 0		; <i1> [#uses=1]
	br i1 %0, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb

; CHECK: bb:
; CHECK: phi i64
; CHECK: sext i8
; CHECK-NOT: sext
bb:		; preds = %bb1, %bb.nph
	%i.02 = phi i32 [ %5, %bb1 ], [ 0, %bb.nph ]		; <i32> [#uses=2]
	%p.01 = phi i8 [ %4, %bb1 ], [ -1, %bb.nph ]		; <i8> [#uses=2]
	%1 = sext i8 %p.01 to i32		; <i32> [#uses=1]
	%2 = sext i32 %i.02 to i64		; <i64> [#uses=1]
	%3 = getelementptr i32, i32* %d, i64 %2		; <i32*> [#uses=1]
	store i32 %1, i32* %3, align 4
	%4 = add i8 %p.01, 1		; <i8> [#uses=1]
	%5 = add i32 %i.02, 1		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%6 = icmp slt i32 %5, %n		; <i1> [#uses=1]
	br i1 %6, label %bb, label %bb1.return_crit_edge

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}
