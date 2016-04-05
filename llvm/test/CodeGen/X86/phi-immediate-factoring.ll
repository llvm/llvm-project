; REQUIRES: asserts
; RUN: llc < %s -disable-preheader-prot=true  -march=x86 -stats 2>&1 | grep "Number of blocks eliminated" | grep 6
; RUN: llc < %s -disable-preheader-prot=false -march=x86 -stats 2>&1 | grep "Number of blocks eliminated" | grep 3
; PR1296

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"

define i32 @foo(i32 %A, i32 %B, i32 %C) nounwind {
entry:
	switch i32 %A, label %out [
		 i32 1, label %bb
		 i32 0, label %bb13
		 i32 2, label %bb35
	]

bb:		; preds = %cond_next, %entry
	%i.144.1 = phi i32 [ 0, %entry ], [ %tmp7, %cond_next ]		; <i32> [#uses=2]
	%tmp4 = and i32 %i.144.1, %B		; <i32> [#uses=1]
	icmp eq i32 %tmp4, 0		; <i1>:0 [#uses=1]
	br i1 %0, label %cond_next, label %out

cond_next:		; preds = %bb
	%tmp7 = add i32 %i.144.1, 1		; <i32> [#uses=2]
	icmp slt i32 %tmp7, 1000		; <i1>:1 [#uses=1]
	br i1 %1, label %bb, label %out

bb13:		; preds = %cond_next18, %entry
	%i.248.1 = phi i32 [ 0, %entry ], [ %tmp20, %cond_next18 ]		; <i32> [#uses=2]
	%tmp16 = and i32 %i.248.1, %C		; <i32> [#uses=1]
	icmp eq i32 %tmp16, 0		; <i1>:2 [#uses=1]
	br i1 %2, label %cond_next18, label %out

cond_next18:		; preds = %bb13
	%tmp20 = add i32 %i.248.1, 1		; <i32> [#uses=2]
	icmp slt i32 %tmp20, 1000		; <i1>:3 [#uses=1]
	br i1 %3, label %bb13, label %out

bb27:		; preds = %bb35
	%tmp30 = and i32 %i.3, %C		; <i32> [#uses=1]
	icmp eq i32 %tmp30, 0		; <i1>:4 [#uses=1]
	br i1 %4, label %cond_next32, label %out

cond_next32:		; preds = %bb27
	%indvar.next = add i32 %i.3, 1		; <i32> [#uses=1]
	br label %bb35

bb35:		; preds = %entry, %cond_next32
	%i.3 = phi i32 [ %indvar.next, %cond_next32 ], [ 0, %entry ]		; <i32> [#uses=3]
	icmp slt i32 %i.3, 1000		; <i1>:5 [#uses=1]
	br i1 %5, label %bb27, label %out

out:		; preds = %bb27, %bb35, %bb13, %cond_next18, %bb, %cond_next, %entry
	%result.0 = phi i32 [ 0, %entry ], [ 1, %bb ], [ 0, %cond_next ], [ 1, %bb13 ], [ 0, %cond_next18 ], [ 1, %bb27 ], [ 0, %bb35 ]		; <i32> [#uses=1]
	ret i32 %result.0
}
