; RUN: llc -verify-machineinstrs < %s -march=ppc32 -enable-tail-merge=0 | grep bl.*baz | count 2
; RUN: llc -verify-machineinstrs < %s -march=ppc32 -enable-tail-merge=0 | grep bl.*quux | count 2
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep bl.*baz | count 1
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep bl.*quux | count 1
; Check that tail merging is the default on ppc, and that -enable-tail-merge works.

; ModuleID = 'tail.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"

define i32 @f(i32 %i, i32 %q) {
entry:
	%i_addr = alloca i32		; <i32*> [#uses=2]
	%q_addr = alloca i32		; <i32*> [#uses=2]
	%retval = alloca i32, align 4		; <i32*> [#uses=1]
	store i32 %i, i32* %i_addr
	store i32 %q, i32* %q_addr
	%tmp = load i32, i32* %i_addr		; <i32> [#uses=1]
	%tmp1 = icmp ne i32 %tmp, 0		; <i1> [#uses=1]
	%tmp12 = zext i1 %tmp1 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp12, 0		; <i1> [#uses=1]
	br i1 %toBool, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	%tmp3 = call i32 (...) @bar( )		; <i32> [#uses=0]
	%tmp4 = call i32 (...) @baz( i32 5, i32 6 )		; <i32> [#uses=0]
	%tmp7 = load i32, i32* %q_addr		; <i32> [#uses=1]
	%tmp8 = icmp ne i32 %tmp7, 0		; <i1> [#uses=1]
	%tmp89 = zext i1 %tmp8 to i8		; <i8> [#uses=1]
	%toBool10 = icmp ne i8 %tmp89, 0		; <i1> [#uses=1]
	br i1 %toBool10, label %cond_true11, label %cond_false15

cond_false:		; preds = %entry
	%tmp5 = call i32 (...) @foo( )		; <i32> [#uses=0]
	%tmp6 = call i32 (...) @baz( i32 5, i32 6 )		; <i32> [#uses=0]
	%tmp27 = load i32, i32* %q_addr		; <i32> [#uses=1]
	%tmp28 = icmp ne i32 %tmp27, 0		; <i1> [#uses=1]
	%tmp289 = zext i1 %tmp28 to i8		; <i8> [#uses=1]
	%toBool210 = icmp ne i8 %tmp289, 0		; <i1> [#uses=1]
	br i1 %toBool210, label %cond_true11, label %cond_false15

cond_true11:		; preds = %cond_next
	%tmp13 = call i32 (...) @foo( )		; <i32> [#uses=0]
	%tmp14 = call i32 (...) @quux( i32 3, i32 4 )		; <i32> [#uses=0]
	br label %cond_next18

cond_false15:		; preds = %cond_next
	%tmp16 = call i32 (...) @bar( )		; <i32> [#uses=0]
	%tmp17 = call i32 (...) @quux( i32 3, i32 4 )		; <i32> [#uses=0]
	br label %cond_next18

cond_next18:		; preds = %cond_false15, %cond_true11
	%tmp19 = call i32 (...) @bar( )		; <i32> [#uses=0]
	br label %return

return:		; preds = %cond_next18
	%retval20 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval20
}

declare i32 @bar(...)

declare i32 @baz(...)

declare i32 @foo(...)

declare i32 @quux(...)
