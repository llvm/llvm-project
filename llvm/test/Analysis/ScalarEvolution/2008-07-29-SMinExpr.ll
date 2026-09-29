; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0 2>&1 | FileCheck %s
; PR2607

define i32 @b(i32 %x, i32 %y) nounwind {
entry:
	%cmp2 = icmp slt i32 %y, %x
	%cond3 = select i1 %cmp2, i32 %y, i32 %x
	%cmp54 = icmp slt i32 %cond3, -2147483632
	br i1 %cmp54, label %forinc, label %afterfor

forinc:		; preds = %forinc, %entry
	%j.01 = phi i32 [ %dec, %forinc ], [ -2147483632, %entry ]
	%dec = add i32 %j.01, -1
	%cmp = icmp slt i32 %y, %x
	%cond = select i1 %cmp, i32 %y, i32 %x
	%cmp5 = icmp sgt i32 %dec, %cond
	br i1 %cmp5, label %forinc, label %afterfor

afterfor:		; preds = %forinc, %entry
	%j.0.lcssa = phi i32 [ -2147483632, %entry ], [ %dec, %forinc ]
	ret i32 %j.0.lcssa
}

; CHECK: backedge-taken count is (-2147483633 + (-1 * (%{{[xy]}} smin %{{[xy]}})))

