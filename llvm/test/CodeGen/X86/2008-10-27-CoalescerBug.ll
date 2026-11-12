; REQUIRES: asserts
; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=+sse2 -stats 2>&1 | FileCheck %s
; Now this test spills one register. But a reload in the loop is cheaper than
; the divsd so it's a win.

define fastcc void @fourn(ptr %data, i32 %isign) nounwind {
; CHECK: fourn
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%indvar93 = phi i32 [ 0, %entry ], [ %idim.030, %bb ]		; <i32> [#uses=2]
	%idim.030 = add i32 %indvar93, 1		; <i32> [#uses=1]
	%0 = add i32 %indvar93, 2		; <i32> [#uses=1]
	%1 = icmp sgt i32 %0, 2		; <i1> [#uses=1]
	br i1 %1, label %bb30.loopexit, label %bb

; CHECK: %bb30.loopexit
; CHECK: divsd %xmm0
; CHECK: movsd %xmm0, 16(%esp)
; CHECK: %bb3
bb3:		; preds = %bb30.loopexit, %bb25, %bb3
	%2 = load i32, ptr null, align 4		; <i32> [#uses=1]
	%3 = mul i32 %2, 0		; <i32> [#uses=1]
	%4 = icmp slt i32 0, %3		; <i1> [#uses=1]
	br i1 %4, label %bb18, label %bb3

bb18:		; preds = %bb3
	%5 = fdiv double %11, 0.000000e+00		; <double> [#uses=1]
	%6 = tail call double @sin(double %5) nounwind readonly		; <double> [#uses=1]
	br label %bb24.preheader

bb22.preheader:		; preds = %bb24.preheader, %bb22.preheader
	br label %bb22.preheader

bb25:		; preds = %bb24.preheader
	%7 = fmul double 0.000000e+00, %6		; <double> [#uses=0]
	%8 = add i32 %i3.122100, 0		; <i32> [#uses=1]
	%9 = icmp sgt i32 %8, 0		; <i1> [#uses=1]
	br i1 %9, label %bb3, label %bb24.preheader

bb24.preheader:		; preds = %bb25, %bb18
	%i3.122100 = or i32 0, 1		; <i32> [#uses=2]
	%10 = icmp slt i32 0, %i3.122100		; <i1> [#uses=1]
	br i1 %10, label %bb25, label %bb22.preheader

bb30.loopexit:		; preds = %bb
	%11 = fmul double 0.000000e+00, 0x401921FB54442D1C		; <double> [#uses=1]
	br label %bb3
}

declare double @sin(double) nounwind readonly
