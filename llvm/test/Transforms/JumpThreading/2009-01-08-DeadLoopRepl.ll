; RUN: llvm-as < %s | opt -jump-threading | llvm-dis
; PR3298

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define i32 @func(i32 %p_79, i32 %p_80) nounwind {
entry:
	br label %bb7

bb1:		; preds = %bb2
	br label %bb2

bb2:		; preds = %bb7, %bb1
	%l_82.0 = phi i8 [ 0, %bb1 ], [ %l_82.1, %bb7 ]		; <i8> [#uses=3]
	br i1 true, label %bb3, label %bb1

bb3:		; preds = %bb2
	%0 = icmp eq i32 %p_80_addr.1, 0		; <i1> [#uses=1]
	br i1 %0, label %bb7, label %bb6

bb5:		; preds = %bb6
	%1 = icmp eq i8 %l_82.0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1.i, label %bb.i

bb.i:		; preds = %bb5
	br label %safe_div_func_char_s_s.exit

bb1.i:		; preds = %bb5
	br label %safe_div_func_char_s_s.exit

safe_div_func_char_s_s.exit:		; preds = %bb1.i, %bb.i
	br label %bb6

bb6:		; preds = %safe_div_func_char_s_s.exit, %bb3
	%p_80_addr.0 = phi i32 [ %p_80_addr.1, %bb3 ], [ 1, %safe_div_func_char_s_s.exit ]		; <i32> [#uses=2]
	%2 = icmp eq i32 %p_80_addr.0, 0		; <i1> [#uses=1]
	br i1 %2, label %bb7, label %bb5

bb7:		; preds = %bb6, %bb3, %entry
	%l_82.1 = phi i8 [ 1, %entry ], [ %l_82.0, %bb3 ], [ %l_82.0, %bb6 ]		; <i8> [#uses=2]
	%p_80_addr.1 = phi i32 [ 0, %entry ], [ %p_80_addr.1, %bb3 ], [ %p_80_addr.0, %bb6 ]		; <i32> [#uses=4]
	%3 = icmp eq i32 %p_80_addr.1, 0		; <i1> [#uses=1]
	br i1 %3, label %bb8, label %bb2

bb8:		; preds = %bb7
	%4 = sext i8 %l_82.1 to i32		; <i32> [#uses=0]
	ret i32 0
}
