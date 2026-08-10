; RUN: opt < %s -passes=gvn | llvm-dis
; PR3358
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.re_pattern_buffer = type { ptr, i64, i64, i64, ptr, ptr, i64, i8 }
	%struct.re_registers = type { i32, ptr, ptr }

define fastcc i32 @byte_re_match_2_internal(ptr nocapture %bufp, ptr %string1, i32 %size1, ptr %string2, i32 %size2, i32 %pos, ptr %regs, i32 %stop) nounwind {
entry:
	br label %bb159

succeed_label:		; preds = %bb159
	ret i32 0

bb159:		; preds = %bb664, %bb554, %bb159, %bb159, %bb159, %entry
	%d.0 = phi ptr [ null, %entry ], [ %d.0, %bb159 ], [ %d.0, %bb554 ], [ %d.0, %bb159 ], [ %d.0, %bb159 ], [ %d.12, %bb664 ]		; <ptr> [#uses=5]
	switch i32 0, label %bb661 [
		i32 0, label %bb159
		i32 1, label %succeed_label
		i32 13, label %bb159
		i32 14, label %bb159
		i32 16, label %bb411
		i32 24, label %bb622
		i32 28, label %bb543
	]

bb411:		; preds = %bb411, %bb159
	br label %bb411

bb543:		; preds = %bb159
	br i1 false, label %bb549, label %bb550

bb549:		; preds = %bb543
	br label %bb554

bb550:		; preds = %bb543
	br i1 false, label %bb554, label %bb552

bb552:		; preds = %bb550
	%0 = load i8, ptr %d.0, align 8		; <i8> [#uses=0]
	br label %bb554

bb554:		; preds = %bb552, %bb550, %bb549
	br i1 false, label %bb159, label %bb661

bb622:		; preds = %bb622, %bb159
	br label %bb622

bb661:		; preds = %bb554, %bb159
	%d.12 = select i1 false, ptr null, ptr null		; <ptr> [#uses=1]
	br label %bb664

bb664:		; preds = %bb664, %bb661
	br i1 false, label %bb159, label %bb664
}
