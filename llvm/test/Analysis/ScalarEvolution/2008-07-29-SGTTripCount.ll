; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution -scalar-evolution-max-iterations=0 | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0 2>&1 | FileCheck %s
; PR2607

define i32 @_Z1aj(i32 %j) nounwind  {
entry:
	icmp sgt i32 0, %j		; <i1>:0 [#uses=1]
	br i1 %0, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	br label %bb

bb:		; preds = %bb, %bb.preheader
	%i.01 = phi i32 [ %1, %bb ], [ 0, %bb.preheader ]		; <i32> [#uses=1]
	add i32 %i.01, -1		; <i32>:1 [#uses=3]
	icmp sgt i32 %1, %j		; <i1>:2 [#uses=1]
	br i1 %2, label %bb, label %return.loopexit

return.loopexit:		; preds = %bb
	br label %return

return:		; preds = %return.loopexit, %entry
	%i.0.lcssa = phi i32 [ 0, %entry ], [ %1, %return.loopexit ]		; <i32> [#uses=1]
	ret i32 %i.0.lcssa
}

; CHECK: backedge-taken count is (-1 + (-1 * %j))

