; RUN: llvm-as < %s | opt -condprop | llvm-dis
; PR3405
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	br label %bb2

bb2:		; preds = %bb.bb2_crit_edge, %entry
	br i1 false, label %bb5.thread2, label %bb

bb:		; preds = %bb2
	br i1 false, label %bb3, label %bb.bb2_crit_edge

bb.bb2_crit_edge:		; preds = %bb
	br label %bb2

bb3:		; preds = %bb
	%.lcssa4 = phi i1 [ false, %bb ]		; <i1> [#uses=1]
	br i1 %.lcssa4, label %bb5.thread, label %bb6

bb5.thread:		; preds = %bb3
	br label %bb7

bb7:		; preds = %bb5.thread2, %bb5.thread
	br label %UnifiedReturnBlock

bb6:		; preds = %bb3
	br label %UnifiedReturnBlock

bb5.thread2:		; preds = %bb2
	br label %bb7

UnifiedReturnBlock:		; preds = %bb6, %bb7
	ret i32 0
}
