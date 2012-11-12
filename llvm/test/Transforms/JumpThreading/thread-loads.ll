; RUN: opt < %s -jump-threading -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

; Test that we can thread through the block with the partially redundant load (%2).
; rdar://6402033
define i32 @test1(i32* %P) nounwind {
; CHECK: @test1
entry:
	%0 = tail call i32 (...)* @f1() nounwind		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1, label %bb

bb:		; preds = %entry
; CHECK: bb1.thread:
; CHECK: store
; CHECK: br label %bb3
	store i32 42, i32* %P, align 4
	br label %bb1

bb1:		; preds = %entry, %bb
	%res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]		; <i32> [#uses=2]
	%2 = load i32* %P, align 4		; <i32> [#uses=1]
	%3 = icmp sgt i32 %2, 36		; <i1> [#uses=1]
	br i1 %3, label %bb3, label %bb2

bb2:		; preds = %bb1
	%4 = tail call i32 (...)* @f2() nounwind		; <i32> [#uses=0]
	ret i32 %res.0

bb3:		; preds = %bb1
; CHECK: bb3:
; CHECK: %res.01 = phi i32 [ 1, %bb1.thread ], [ 0, %bb1 ]
; CHECK: ret i32 %res.01
	ret i32 %res.0
}

declare i32 @f1(...)

declare i32 @f2(...)


;; Check that we preserve TBAA information.
; rdar://11039258

define i32 @test2(i32* %P) nounwind {
; CHECK: @test2
entry:
	%0 = tail call i32 (...)* @f1() nounwind		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1, label %bb

bb:		; preds = %entry
; CHECK: bb1.thread:
; CHECK: store{{.*}}, !tbaa !0
; CHECK: br label %bb3
	store i32 42, i32* %P, align 4, !tbaa !0
	br label %bb1

bb1:		; preds = %entry, %bb
	%res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
	%2 = load i32* %P, align 4, !tbaa !0
	%3 = icmp sgt i32 %2, 36
	br i1 %3, label %bb3, label %bb2

bb2:		; preds = %bb1
	%4 = tail call i32 (...)* @f2() nounwind
	ret i32 %res.0

bb3:		; preds = %bb1
; CHECK: bb3:
; CHECK: %res.01 = phi i32 [ 1, %bb1.thread ], [ 0, %bb1 ]
; CHECK: ret i32 %res.01
	ret i32 %res.0
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}

