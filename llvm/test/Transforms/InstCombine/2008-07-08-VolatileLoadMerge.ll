; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {volatile load} | count 2
; PR2496
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@g_1 = internal global i32 0		; <i32*> [#uses=3]

define i32 @main() nounwind  {
entry:
	%tmp93 = icmp slt i32 0, 10		; <i1> [#uses=0]
	%tmp34 = volatile load i32* @g_1, align 4		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %entry
	%b.0.reg2mem.0 = phi i32 [ 0, %entry ], [ %tmp6, %bb ]		; <i32> [#uses=1]
	%tmp3.reg2mem.0 = phi i32 [ %tmp3, %bb ], [ %tmp34, %entry ]
	%tmp4 = add i32 %tmp3.reg2mem.0, 5		; <i32> [#uses=1]
	volatile store i32 %tmp4, i32* @g_1, align 4
	%tmp6 = add i32 %b.0.reg2mem.0, 1		; <i32> [#uses=2]
	%tmp9 = icmp slt i32 %tmp6, 10		; <i1> [#uses=1]
	%tmp3 = volatile load i32* @g_1, align 4		; <i32> [#uses=1]
	br i1 %tmp9, label %bb, label %bb11

bb11:		; preds = %bb
	ret i32 0
}

