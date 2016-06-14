; RUN: opt < %s -jump-threading -S | FileCheck %s
; RUN: opt < %s -passes=jump-threading -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

; Test that we can thread through the block with the partially redundant load (%2).
; rdar://6402033
define i32 @test1(i32* %P) nounwind {
; CHECK-LABEL: @test1(
entry:
	%0 = tail call i32 (...) @f1() nounwind		; <i32> [#uses=1]
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
	%2 = load i32, i32* %P, align 4		; <i32> [#uses=1]
	%3 = icmp sgt i32 %2, 36		; <i1> [#uses=1]
	br i1 %3, label %bb3, label %bb2

bb2:		; preds = %bb1
	%4 = tail call i32 (...) @f2() nounwind		; <i32> [#uses=0]
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
; CHECK-LABEL: @test2(
entry:
	%0 = tail call i32 (...) @f1() nounwind		; <i32> [#uses=1]
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
	%2 = load i32, i32* %P, align 4, !tbaa !0
	%3 = icmp sgt i32 %2, 36
	br i1 %3, label %bb3, label %bb2

bb2:		; preds = %bb1
	%4 = tail call i32 (...) @f2() nounwind
	ret i32 %res.0

bb3:		; preds = %bb1
; CHECK: bb3:
; CHECK: %res.01 = phi i32 [ 1, %bb1.thread ], [ 0, %bb1 ]
; CHECK: ret i32 %res.01
	ret i32 %res.0
}

define i32 @test3(i8** %x, i1 %f) {
; Correctly thread loads of different (but compatible) types, placing bitcasts
; as necessary in the predecessors. This is especially tricky because the same
; predecessor ends up with two entries in the PHI node and they must share
; a single cast.
; CHECK-LABEL: @test3(
entry:
  %0 = bitcast i8** %x to i32**
  %1 = load i32*, i32** %0, align 8
  br i1 %f, label %if.end57, label %if.then56
; CHECK: %[[LOAD:.*]] = load i32*, i32**
; CHECK: %[[CAST:.*]] = bitcast i32* %[[LOAD]] to i8*

if.then56:
  br label %if.end57

if.end57:
  %2 = load i8*, i8** %x, align 8
  %tobool59 = icmp eq i8* %2, null
  br i1 %tobool59, label %return, label %if.then60
; CHECK: %[[PHI:.*]] = phi i8* [ %[[CAST]], %[[PRED:[^ ]+]] ], [ %[[CAST]], %[[PRED]] ]
; CHECK-NEXT: %[[CMP:.*]] = icmp eq i8* %[[PHI]], null
; CHECK-NEXT: br i1 %[[CMP]]

if.then60:
  ret i32 42

return:
  ret i32 13
}

!0 = !{!3, !3, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA", null}
!3 = !{!"int", !1}
