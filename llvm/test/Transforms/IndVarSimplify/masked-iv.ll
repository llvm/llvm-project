; RUN: opt < %s -passes=indvars -S | FileCheck %s

; Indvars should do the IV arithmetic in the canonical IV type (i64),
; and only use one truncation.

define void @foo(ptr %A, ptr %B, i64 %n, i64 %a, i64 %s) nounwind {
; CHECK-LABEL: @foo(
; CHECK-NOT: trunc
; CHECK: and
; CHECK-NOT: and
entry:
	%t0 = icmp sgt i64 %n, 0		; <i1> [#uses=1]
	br i1 %t0, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	br label %bb

bb:		; preds = %bb, %bb.preheader
	%i.01 = phi i64 [ %t6, %bb ], [ %a, %bb.preheader ]		; <i64> [#uses=3]
	%t1 = and i64 %i.01, 255		; <i64> [#uses=1]
	%t2 = getelementptr i64, ptr %A, i64 %t1		; <ptr> [#uses=1]
	store i64 %i.01, ptr %t2, align 8
	%t6 = add i64 %i.01, %s		; <i64> [#uses=1]
	br label %bb

return:		; preds = %entry
	ret void
}
