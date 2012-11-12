; RUN: llc < %s -o - | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9"

; The comparison happens before the relevant use, but it can still be rewritten
; to compare with zero.

; CHECK: foo:
; CHECK: align
; CHECK: incl  %eax
; CHECK-NEXT: decl  %ecx
; CHECK-NEXT: jne

define void @foo() nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i32 [ 0, %entry ], [ %i.2.0.us1534, %loop ]		; <i32> [#uses=1]
	%i.2.0.us1534 = add i32 %indvar, 1		; <i32> [#uses=3]
	%tmp611.us1535 = icmp eq i32 %i.2.0.us1534, 4		; <i1> [#uses=2]
	%tmp623.us1538 = select i1 %tmp611.us1535, i32 6, i32 0		; <i32> [#uses=0]
	%tmp628.us1540 = shl i32 %i.2.0.us1534, 1		; <i32> [#uses=1]
	%tmp645646647.us1547 = sext i32 %tmp628.us1540 to i64		; <i64> [#uses=0]
	br i1 %tmp611.us1535, label %exit, label %loop

exit:
	ret void
}
