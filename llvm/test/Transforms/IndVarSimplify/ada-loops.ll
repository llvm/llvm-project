; RUN: opt < %s -indvars -S | FileCheck %s
;
; PR1301

; Do a bunch of analysis and prove that the loops can use an i32 trip
; count without casting.
;
; Note that all four functions should actually be converted to
; memset. However, this test case validates indvars behavior.  We
; don't check that phis are "folded together" because that is a job
; for loop strength reduction. But indvars must remove sext, zext, and add i8.
;
; CHECK-NOT: {{sext|zext|add i8}}

; ModuleID = 'ada.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32"
target triple = "i686-pc-linux-gnu"

define void @kinds__sbytezero([256 x i32]* nocapture %a) nounwind {
bb.thread:
	%tmp46 = getelementptr [256 x i32], [256 x i32]* %a, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %tmp46
	br label %bb

bb:		; preds = %bb, %bb.thread
	%i.0.reg2mem.0 = phi i8 [ -128, %bb.thread ], [ %tmp8, %bb ]		; <i8> [#uses=1]
	%tmp8 = add i8 %i.0.reg2mem.0, 1		; <i8> [#uses=3]
	%tmp1 = sext i8 %tmp8 to i32		; <i32> [#uses=1]
	%tmp3 = add i32 %tmp1, 128		; <i32> [#uses=1]
	%tmp4 = getelementptr [256 x i32], [256 x i32]* %a, i32 0, i32 %tmp3		; <i32*> [#uses=1]
	store i32 0, i32* %tmp4
	%0 = icmp eq i8 %tmp8, 127		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb

return:		; preds = %bb
	ret void
}

define void @kinds__ubytezero([256 x i32]* nocapture %a) nounwind {
bb.thread:
	%tmp35 = getelementptr [256 x i32], [256 x i32]* %a, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %tmp35
	br label %bb

bb:		; preds = %bb, %bb.thread
	%i.0.reg2mem.0 = phi i8 [ 0, %bb.thread ], [ %tmp7, %bb ]		; <i8> [#uses=1]
	%tmp7 = add i8 %i.0.reg2mem.0, 1		; <i8> [#uses=3]
	%tmp1 = zext i8 %tmp7 to i32		; <i32> [#uses=1]
	%tmp3 = getelementptr [256 x i32], [256 x i32]* %a, i32 0, i32 %tmp1		; <i32*> [#uses=1]
	store i32 0, i32* %tmp3
	%0 = icmp eq i8 %tmp7, -1		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb

return:		; preds = %bb
	ret void
}

define void @kinds__srangezero([21 x i32]* nocapture %a) nounwind {
bb.thread:
	br label %bb

bb:		; preds = %bb, %bb.thread
	%i.0.reg2mem.0 = phi i8 [ -10, %bb.thread ], [ %tmp7, %bb ]		; <i8> [#uses=2]
	%tmp12 = sext i8 %i.0.reg2mem.0 to i32		; <i32> [#uses=1]
	%tmp4 = add i32 %tmp12, 10		; <i32> [#uses=1]
	%tmp5 = getelementptr [21 x i32], [21 x i32]* %a, i32 0, i32 %tmp4		; <i32*> [#uses=1]
	store i32 0, i32* %tmp5
	%tmp7 = add i8 %i.0.reg2mem.0, 1		; <i8> [#uses=2]
	%0 = icmp sgt i8 %tmp7, 10		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb

return:		; preds = %bb
	ret void
}

define void @kinds__urangezero([21 x i32]* nocapture %a) nounwind {
bb.thread:
	br label %bb

bb:		; preds = %bb, %bb.thread
	%i.0.reg2mem.0 = phi i8 [ 10, %bb.thread ], [ %tmp7, %bb ]		; <i8> [#uses=2]
	%tmp12 = sext i8 %i.0.reg2mem.0 to i32		; <i32> [#uses=1]
	%tmp4 = add i32 %tmp12, -10		; <i32> [#uses=1]
	%tmp5 = getelementptr [21 x i32], [21 x i32]* %a, i32 0, i32 %tmp4		; <i32*> [#uses=1]
	store i32 0, i32* %tmp5
	%tmp7 = add i8 %i.0.reg2mem.0, 1		; <i8> [#uses=2]
	%0 = icmp sgt i8 %tmp7, 30		; <i1> [#uses=1]
	br i1 %0, label %return, label %bb

return:		; preds = %bb
	ret void
}
