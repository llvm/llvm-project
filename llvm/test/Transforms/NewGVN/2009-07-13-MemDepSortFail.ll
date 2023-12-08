; RUN: opt < %s -passes=newgvn | llvm-dis
; PR4256
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%llvm.dbg.anchor.type = type { i32, i32 }
	%struct.cset = type { ptr, i8, i8, i32, ptr }
	%struct.lmat = type { ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr }
	%struct.re_guts = type { ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, i32, i32, i32, [1 x i8] }

define ptr @lbackref(ptr %m, ptr %start, ptr %stop, i32 %startst, i32 %stopst, i32 %lev, i32 %rec) nounwind {
entry:
	br label %bb63

bb:		; preds = %bb63
	switch i32 0, label %bb62 [
		i32 268435456, label %bb2
		i32 805306368, label %bb9
		i32 -1610612736, label %bb51
	]

bb2:		; preds = %bb
	br label %bb62

bb9:		; preds = %bb
	%0 = load i8, ptr %sp.1, align 1		; <i8> [#uses=0]
	br label %bb62

bb51:		; preds = %bb
	%1 = load i8, ptr %sp.1, align 1		; <i8> [#uses=0]
	ret ptr null

bb62:		; preds = %bb9, %bb2, %bb
	br label %bb63

bb63:		; preds = %bb84, %bb69, %bb62, %entry
	%sp.1 = phi ptr [ null, %bb62 ], [ %sp.1.lcssa, %bb84 ], [ %start, %entry ], [ %sp.1.lcssa, %bb69 ]		; <ptr> [#uses=3]
	br i1 false, label %bb, label %bb65

bb65:		; preds = %bb63
	%sp.1.lcssa = phi ptr [ %sp.1, %bb63 ]		; <ptr> [#uses=4]
	br i1 false, label %bb66, label %bb69

bb66:		; preds = %bb65
	ret ptr null

bb69:		; preds = %bb65
	switch i32 0, label %bb108.loopexit2.loopexit.loopexit [
		i32 1342177280, label %bb63
		i32 1476395008, label %bb84
		i32 1879048192, label %bb104
		i32 2013265920, label %bb93
	]

bb84:		; preds = %bb69
	%2 = tail call ptr @lbackref(ptr %m, ptr %sp.1.lcssa, ptr %stop, i32 0, i32 %stopst, i32 0, i32 0) nounwind		; <ptr> [#uses=0]
	br label %bb63

bb93:		; preds = %bb69
	ret ptr null

bb104:		; preds = %bb69
	%sp.1.lcssa.lcssa33 = phi ptr [ %sp.1.lcssa, %bb69 ]		; <ptr> [#uses=0]
	unreachable

bb108.loopexit2.loopexit.loopexit:		; preds = %bb69
	ret ptr null
}
