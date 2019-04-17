; RUN: opt < %s -instcombine -S | grep "load volatile" | count 2
; PR2262
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@g_1 = internal global i32 0		; <i32*> [#uses=3]

define i32 @main(i32 %i) nounwind  {
entry:
	%tmp93 = icmp slt i32 %i, 10		; <i1> [#uses=0]
	%tmp34 = load volatile i32, i32* @g_1, align 4		; <i32> [#uses=1]
	br i1 %tmp93, label %bb11, label %bb

bb:		; preds = %bb, %entry
	%tmp3 = load volatile i32, i32* @g_1, align 4		; <i32> [#uses=1]
	br label %bb11

bb11:		; preds = %bb
	%tmp4 = phi i32 [ %tmp34, %entry ], [ %tmp3, %bb ]		; <i32> [#uses=1]
	ret i32 %tmp4
}

