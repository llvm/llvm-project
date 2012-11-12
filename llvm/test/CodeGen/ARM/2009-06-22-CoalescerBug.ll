; RUN: llc < %s -mtriple=armv6-apple-darwin

	%struct.rtunion = type { i64 }
	%struct.rtx_def = type { i16, i8, i8, [1 x %struct.rtunion] }

define void @simplify_unary_real(i8* nocapture %p) nounwind {
entry:
	%tmp121 = load i64* null, align 4		; <i64> [#uses=1]
	%0 = getelementptr %struct.rtx_def* null, i32 0, i32 3, i32 3, i32 0		; <i64*> [#uses=1]
	%tmp122 = load i64* %0, align 4		; <i64> [#uses=1]
	%1 = zext i64 undef to i192		; <i192> [#uses=2]
	%2 = zext i64 %tmp121 to i192		; <i192> [#uses=1]
	%3 = shl i192 %2, 64		; <i192> [#uses=2]
	%4 = zext i64 %tmp122 to i192		; <i192> [#uses=1]
	%5 = shl i192 %4, 128		; <i192> [#uses=1]
	%6 = or i192 %3, %1		; <i192> [#uses=1]
	%7 = or i192 %6, %5		; <i192> [#uses=2]
	switch i32 undef, label %bb82 [
		i32 77, label %bb38
		i32 129, label %bb21
		i32 130, label %bb20
	]

bb20:		; preds = %entry
	ret void

bb21:		; preds = %entry
	br i1 undef, label %bb82, label %bb29

bb29:		; preds = %bb21
	%tmp18.i = and i192 %3, 1208907372870555465154560		; <i192> [#uses=1]
	%mask.i = or i192 %tmp18.i, %1		; <i192> [#uses=1]
	%mask41.i = or i192 %mask.i, 0		; <i192> [#uses=1]
	br label %bb82

bb38:		; preds = %entry
	br label %bb82

bb82:		; preds = %bb38, %bb29, %bb21, %entry
	%d.0 = phi i192 [ %mask41.i, %bb29 ], [ undef, %bb38 ], [ %7, %entry ], [ %7, %bb21 ]		; <i192> [#uses=1]
	%tmp51 = trunc i192 %d.0 to i64		; <i64> [#uses=0]
	ret void
}
