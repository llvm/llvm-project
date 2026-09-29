; RUN: llc -mtriple=i686-- < %s | FileCheck %s

; The comparison happens after the relevant use, so the stride can easily
; be changed. The comparison can be done in a narrower mode than the
; induction variable.
; TODO: By making the first store post-increment as well, the loop setup
; could be made simpler.

define void @foo() nounwind {
; CHECK-LABEL: foo:
; CHECK-NOT: ret
; CHECK: cmpl $10
; CHECK: ret

entry:
	br label %loop

loop:
	%indvar = phi i32 [ 0, %entry ], [ %i.2.0.us1534, %loop ]		; <i32> [#uses=1]
	%i.2.0.us1534 = add i32 %indvar, 1		; <i32> [#uses=3]
	%tmp628.us1540 = shl i32 %i.2.0.us1534, 1		; <i32> [#uses=1]
	%tmp645646647.us1547 = sext i32 %tmp628.us1540 to i64		; <i64> [#uses=1]
	store i64 %tmp645646647.us1547, ptr null
	%tmp611.us1535 = icmp eq i32 %i.2.0.us1534, 4		; <i1> [#uses=2]
	%tmp623.us1538 = select i1 %tmp611.us1535, i32 6, i32 0		; <i32> [#uses=1]
	store i32 %tmp623.us1538, ptr null
	br i1 %tmp611.us1535, label %exit, label %loop

exit:
	ret void
}
