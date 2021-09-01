; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s
; PR3275

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'func_15'
; CHECK-NOT: /u -1

@g_16 = external global i16		; <i16*> [#uses=3]
@.str = external constant [4 x i8]		; <[4 x i8]*> [#uses=0]

define void @func_15() nounwind {
entry:
	%0 = load i16, i16* @g_16, align 2		; <i16> [#uses=1]
	%1 = icmp sgt i16 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb2, label %bb.nph

bb.nph:		; preds = %entry
	%g_16.promoted = load i16, i16* @g_16		; <i16> [#uses=1]
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%g_16.tmp.0 = phi i16 [ %g_16.promoted, %bb.nph ], [ %2, %bb1 ]		; <i16> [#uses=1]
	%2 = add i16 %g_16.tmp.0, -1		; <i16> [#uses=3]
	br label %bb1

bb1:		; preds = %bb
	%3 = icmp sgt i16 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %bb1.bb2_crit_edge, label %bb

bb1.bb2_crit_edge:		; preds = %bb1
	store i16 %2, i16* @g_16
	br label %bb2

bb2:		; preds = %bb1.bb2_crit_edge, %entry
	br label %return

return:		; preds = %bb2
	ret void
}

declare i32 @main() nounwind

declare i32 @printf(i8*, ...) nounwind

