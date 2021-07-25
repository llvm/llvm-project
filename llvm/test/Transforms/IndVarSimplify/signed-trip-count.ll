; RUN: opt < %s -indvars -S | FileCheck %s

; Provide legal integer types.
target datalayout = "e-p:32:32:32-n8:16:32:64"


define void @foo(i64* nocapture %x, i32 %n) nounwind {
; CHECK-LABEL: @foo(
; CHECK-NOT: sext
; CHECK: phi
; CHECK-NOT: phi
entry:
	%tmp102 = icmp sgt i32 %n, 0		; <i1> [#uses=1]
	br i1 %tmp102, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb7, %bb.nph
	%i.01 = phi i32 [ %tmp6, %bb7 ], [ 0, %bb.nph ]		; <i32> [#uses=3]
	%tmp1 = sext i32 %i.01 to i64		; <i64> [#uses=1]
	%tmp4 = getelementptr i64, i64* %x, i32 %i.01		; <i64*> [#uses=1]
	store i64 %tmp1, i64* %tmp4, align 8
	%tmp6 = add i32 %i.01, 1		; <i32> [#uses=2]
	br label %bb7

bb7:		; preds = %bb
	%tmp10 = icmp slt i32 %tmp6, %n		; <i1> [#uses=1]
	br i1 %tmp10, label %bb, label %bb7.return_crit_edge

bb7.return_crit_edge:		; preds = %bb7
	br label %return

return:		; preds = %bb7.return_crit_edge, %entry
	ret void
}
