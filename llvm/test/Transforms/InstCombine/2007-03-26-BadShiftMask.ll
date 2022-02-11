; PR1271
; RUN: opt < %s -passes=instcombine -S | \
; RUN:    grep "ashr exact i32 %.mp137, 2"

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"


define i1 @test(i32* %tmp141, i32* %tmp145, 
            i32 %b8, i32 %iftmp.430.0, i32* %tmp134.out, i32* %tmp137.out)
{
newFuncRoot:
	%tmp133 = and i32 %b8, 1		; <i32> [#uses=1]
	%tmp134 = shl i32 %tmp133, 3		; <i32> [#uses=3]
	%tmp136 = ashr i32 %b8, 1		; <i32> [#uses=1]
	%tmp137 = shl i32 %tmp136, 3		; <i32> [#uses=3]
	%tmp139 = ashr i32 %tmp134, 2		; <i32> [#uses=1]
	store i32 %tmp139, i32* %tmp141
	%tmp143 = ashr i32 %tmp137, 2		; <i32> [#uses=1]
	store i32 %tmp143, i32* %tmp145
	icmp eq i32 %iftmp.430.0, 0		; <i1>:0 [#uses=1]
	zext i1 %0 to i8		; <i8>:1 [#uses=1]
	icmp ne i8 %1, 0		; <i1>:2 [#uses=1]
	br i1 %2, label %cond_true147.exitStub, label %cond_false252.exitStub

cond_true147.exitStub:		; preds = %newFuncRoot
	store i32 %tmp134, i32* %tmp134.out
	store i32 %tmp137, i32* %tmp137.out
	ret i1 true

cond_false252.exitStub:		; preds = %newFuncRoot
	store i32 %tmp134, i32* %tmp134.out
	store i32 %tmp137, i32* %tmp137.out
	ret i1 false
}
