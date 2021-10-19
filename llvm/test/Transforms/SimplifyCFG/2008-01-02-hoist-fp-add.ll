; The phi should not be eliminated in this case, because the divide op could trap.
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
@G = weak global i32 0, align 8		; <i32*> [#uses=2]

define void @test(i32 %X, i32 %Y, i32 %Z) {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = load i32, i32* @G, align 8		; <i32> [#uses=2]
	%tmp3 = icmp eq i32 %X, %Y		; <i1> [#uses=1]
	%tmp34 = zext i1 %tmp3 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp34, 0		; <i1> [#uses=1]
	br i1 %toBool, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%tmp7 = udiv i32 %tmp, %Z		; <i32> [#uses=1]
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
; CHECK: = phi i32
	%F.0 = phi i32 [ %tmp, %entry ], [ %tmp7, %cond_true ]		; <i32> [#uses=1]
	store i32 %F.0, i32* @G, align 8
	ret void
}

