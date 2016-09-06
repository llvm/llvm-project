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

define i32 @test4(i32* %P) {
; CHECK-LABEL: @test4(
entry:
  %v0 = tail call i32 (...) @f1()
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %bb1, label %bb

bb:
; CHECK: bb1.thread:
; CHECK: store atomic
; CHECK: br label %bb3
  store atomic i32 42, i32* %P unordered, align 4
  br label %bb1

bb1:
; CHECK: bb1:
; CHECK-NOT: phi
; CHECK: load atomic
  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
  %v2 = load atomic i32, i32* %P unordered, align 4
  %v3 = icmp sgt i32 %v2, 36
  br i1 %v3, label %bb3, label %bb2

bb2:
  %v4 = tail call i32 (...) @f2()
  ret i32 %res.0

bb3:
  ret i32 %res.0
}

define i32 @test5(i32* %P) {
; Negative test

; CHECK-LABEL: @test5(
entry:
  %v0 = tail call i32 (...) @f1()
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %bb1, label %bb

bb:
; CHECK: bb:
; CHECK-NEXT:   store atomic i32 42, i32* %P release, align 4
; CHECK-NEXT:   br label %bb1
  store atomic i32 42, i32* %P release, align 4
  br label %bb1

bb1:
; CHECK: bb1:
; CHECK-NEXT:  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
; CHECK-NEXT:  %v2 = load atomic i32, i32* %P acquire, align 4
; CHECK-NEXT:  %v3 = icmp sgt i32 %v2, 36
; CHECK-NEXT:  br i1 %v3, label %bb3, label %bb2

  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
  %v2 = load atomic i32, i32* %P acquire, align 4
  %v3 = icmp sgt i32 %v2, 36
  br i1 %v3, label %bb3, label %bb2

bb2:
  %v4 = tail call i32 (...) @f2()
  ret i32 %res.0

bb3:
  ret i32 %res.0
}

define i32 @test6(i32* %P) {
; Negative test

; CHECK-LABEL: @test6(
entry:
  %v0 = tail call i32 (...) @f1()
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %bb1, label %bb

bb:
; CHECK: bb:
; CHECK-NEXT:   store i32 42, i32* %P
; CHECK-NEXT:   br label %bb1
  store i32 42, i32* %P
  br label %bb1

bb1:
; CHECK: bb1:
; CHECK-NEXT:  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
; CHECK-NEXT:  %v2 = load atomic i32, i32* %P acquire, align 4
; CHECK-NEXT:  %v3 = icmp sgt i32 %v2, 36
; CHECK-NEXT:  br i1 %v3, label %bb3, label %bb2

  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
  %v2 = load atomic i32, i32* %P acquire, align 4
  %v3 = icmp sgt i32 %v2, 36
  br i1 %v3, label %bb3, label %bb2

bb2:
  %v4 = tail call i32 (...) @f2()
  ret i32 %res.0

bb3:
  ret i32 %res.0
}

define i32 @test7(i32* %P) {
; Negative test

; CHECK-LABEL: @test7(
entry:
  %v0 = tail call i32 (...) @f1()
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %bb1, label %bb

bb:
; CHECK: bb:
; CHECK-NEXT:   %val = load i32, i32* %P
; CHECK-NEXT:   br label %bb1
  %val = load i32, i32* %P
  br label %bb1

bb1:
; CHECK: bb1:
; CHECK-NEXT:  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
; CHECK-NEXT:  %v2 = load atomic i32, i32* %P acquire, align 4
; CHECK-NEXT:  %v3 = icmp sgt i32 %v2, 36
; CHECK-NEXT:  br i1 %v3, label %bb3, label %bb2

  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
  %v2 = load atomic i32, i32* %P acquire, align 4
  %v3 = icmp sgt i32 %v2, 36
  br i1 %v3, label %bb3, label %bb2

bb2:
  %v4 = tail call i32 (...) @f2()
  ret i32 %res.0

bb3:
  ret i32 %res.0
}

; Make sure we merge the aliasing metadata. (If we don't, we have a load
; with the wrong metadata, so the branch gets incorrectly eliminated.)
define void @test8(i32*, i32*, i32*) {
; CHECK-LABEL: @test8(
; CHECK: %a = load i32, i32* %0, !range !4
; CHECK-NEXT: store i32 %a
; CHECK: br i1 %c
  %a = load i32, i32* %0, !tbaa !0, !range !4, !alias.scope !9, !noalias !10
  %b = load i32, i32* %0, !range !5
  store i32 %a, i32* %1
  %c = icmp eq i32 %b, 8
  br i1 %c, label %ret1, label %ret2

ret1:
  ret void

ret2:
  %xxx = tail call i32 (...) @f1() nounwind
  ret void
}

; Make sure we merge/PRE aliasing metadata correctly.  That means that
; we need to remove metadata from the existing load, and add appropriate
; metadata to the newly inserted load.
define void @test9(i32*, i32*, i32*, i1 %c) {
; CHECK-LABEL: @test9(
  br i1 %c, label %d1, label %d2

; CHECK: d1:
; CHECK-NEXT: %a = load i32, i32* %0{{$}}
d1:
  %a = load i32, i32* %0, !range !4, !alias.scope !9, !noalias !10
  br label %d3

; CHECK: d2:
; CHECK-NEXT: %xxxx = tail call i32 (...) @f1()
; CHECK-NEXT: %b.pr = load i32, i32* %0, !tbaa !0{{$}}
d2:
  %xxxx = tail call i32 (...) @f1() nounwind
  br label %d3

d3:
  %p = phi i32 [ 1, %d2 ], [ %a, %d1 ]
  %b = load i32, i32* %0, !tbaa !0
  store i32 %p, i32* %1
  %c2 = icmp eq i32 %b, 8
  br i1 %c2, label %ret1, label %ret2

ret1:
  ret void

ret2:
  %xxx = tail call i32 (...) @f1() nounwind
  ret void
}

!0 = !{!3, !3, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA", null}
!3 = !{!"int", !1}
!4 = !{ i32 0, i32 1 }
!5 = !{ i32 8, i32 10 }
!6 = !{!6}
!7 = !{!7, !6}
!8 = !{!8, !6}
!9 = !{!7}
!10 = !{!8}
