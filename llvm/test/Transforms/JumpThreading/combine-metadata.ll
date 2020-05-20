; RUN: opt < %s -jump-threading -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

declare void @use(i32 *)

; Check that we propagate nonnull to dominated loads, when we find an available
; loaded value.
; CHECK-LABEL: @test1(
; CHECK-LABEL: ret1:
; CHECK-NEXT: %[[p1:.*]] = load i32*, i32** %ptr
; CHECK-NOT: !nonnull
; CHECK-NEXT: store i32 1, i32* %[[p1]]
; CHECK-NEXT: tail call void @use(i32* null)
; CHECK-NEXT: ret void

; CHECK-LABEL: ret2:
; CHECK-NEXT: %[[p2:.*]] = load i32*, i32** %ptr, !nonnull !0
; CHECK: tail call void @use(i32* %[[p2]])
; CHECK-NEXT: ret void
define void @test1(i32** %ptr, i1 %c) {
  br i1 %c, label %d1, label %d2

d1:
  %p1 = load i32*, i32** %ptr, !nonnull !0
  br label %d3

d2:
  br label %d3

d3:
  %pm = phi i32* [ null, %d2 ], [ %p1, %d1 ]
  %p2 = load i32*, i32** %ptr
  store i32 1, i32* %p2
  %c2 = icmp eq i32* %pm, null
  br i1 %c2, label %ret1, label %ret2

ret1:
  tail call void @use(i32* %pm) nounwind
  ret void

ret2:
  tail call void @use(i32* %pm) nounwind
  ret void
}

; Check that we propagate nonnull to dominated loads, when we find an available
; loaded value.
; CHECK-LABEL: @test2(
; CHECK-LABEL: d3.thread:
; CHECK-NEXT: %[[p1:.*]] = load i32*, i32** %ptr, !nonnull !0
; CHECK-NEXT: store i32 1, i32* %[[p1]]
; CHECK-NEXT: br label %ret1

; CHECK-LABEL: d3:
; CHECK-NEXT: %[[p_cmp:.*]] = load i32*, i32** %ptr
; CHECK-NEXT: %[[p2:.*]] = load i32*, i32** %ptr, !nonnull !0
; CHECK-NEXT: store i32 1, i32* %[[p2]]
; CHECK-NEXT: icmp eq i32* %[[p_cmp]], null
define void @test2(i32** %ptr, i1 %c) {
  br i1 %c, label %d1, label %d2

d1:
  %p1 = load i32*, i32** %ptr
  br label %d3

d2:
  br label %d3

d3:
  %pm = phi i32* [ null, %d2 ], [ %p1, %d1 ]
  %p2 = load i32*, i32** %ptr, !nonnull !0
  store i32 1, i32* %p2
  %c2 = icmp eq i32* %pm, null
  br i1 %c2, label %ret1, label %ret2

ret1:
  tail call void @use(i32* %pm) nounwind
  ret void

ret2:
  tail call void @use(i32* %pm) nounwind
  ret void
}

; Check that we do not propagate nonnull to loads predecessors that are combined
; to a PHI node.
; CHECK-LABEL: @test3(
; CHECK-LABEL: d1:
; CHECK-NEXT: %[[p1:.*]] = load i32*, i32** %ptr
; CHECK-NOT: !nonnull

; CHECK-LABEL: d2:
; CHECK-NEXT: %[[p2:.*]] = load i32*, i32** %ptr
; CHECK-NOT: !nonnull

; CHECK-LABEL: d3:
; CHECK-NEXT:  phi i32* [ %[[p2]], %d2 ], [ %[[p1]], %d1 ]
define void @test3(i32** %ptr) {
d1:
  %x = load i32*, i32** %ptr, !nonnull !0
  br label %d3

d2:
  br label %d3

d3:
  %y = load i32*, i32** %ptr
  store i32 1, i32* %y
  %c2 = icmp eq i32* %y, null
  br i1 %c2, label %ret1, label %ret2

ret1:
  ret void

ret2:
  ret void
}


!0 = !{}
