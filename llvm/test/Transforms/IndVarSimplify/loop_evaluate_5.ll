; RUN: opt < %s -indvars -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"

; Indvars should be able to compute an exit value for %tmp1.

define i32 @testcase(i5 zeroext %k) nounwind readnone {
; CHECK-LABEL: @testcase
; CHECK: 120, %bb2.bb3_crit_edge
entry:
	br i1 false, label %bb3, label %bb.nph

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb2, %bb.nph
	%result2 = phi i32 [ %tmp1, %bb2 ], [ 0, %bb.nph ]		; <i32> [#uses=1]
	%k_01 = phi i5 [ %indvar_next1, %bb2 ], [ 0, %bb.nph ]		; <i5> [#uses=2]
	%tmp2 = zext i5 %k_01 to i32		; <i32> [#uses=1]
	%tmp1 = add i32 %tmp2, %result2		; <i32> [#uses=2]
	%indvar_next1 = add i5 %k_01, 1		; <i5> [#uses=2]
	br label %bb2

bb2:		; preds = %bb
	%phitmp = icmp eq i5 %indvar_next1, -16		; <i1> [#uses=1]
	br i1 %phitmp, label %bb2.bb3_crit_edge, label %bb

bb2.bb3_crit_edge:		; preds = %bb2
	br label %bb3

bb3:		; preds = %bb2.bb3_crit_edge, %entry
	%result.lcssa = phi i32 [ %tmp1, %bb2.bb3_crit_edge ], [ 0, %entry ]		; <i32> [#uses=1]
	ret i32 %result.lcssa
}
