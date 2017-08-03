; RUN: llc < %s -march=x86
; PR2098

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @foo(i32 %x) {
entry:
	%x_addr = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %x, i32* %x_addr
	%tmp = load i32, i32* %x_addr, align 4		; <i32> [#uses=1]
	%tmp1 = ashr i32 %tmp, -2		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 1		; <i32> [#uses=1]
	%tmp23 = trunc i32 %tmp2 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp23, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb, label %bb5

bb:		; preds = %entry
	%tmp4 = call i32 (...) @bar( ) nounwind 		; <i32> [#uses=0]
	br label %bb5

bb5:		; preds = %bb, %entry
	br label %return

return:		; preds = %bb5
	ret void
}

declare i32 @bar(...)
