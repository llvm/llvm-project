; RUN: opt -passes=newgvn -S < %s | FileCheck %s

; Check that an unreachable predecessor to a PHI node doesn't cause a crash.
; PR21625.

define i32 @f(i32** %f) {
; CHECK: bb0:
; Load should be removed, since it's ignored.
; CHECK-NEXT: br label
bb0:
  %bar = load i32*, i32** %f
  br label %bb2
bb1:
  %zed = load i32*, i32** %f
  br i1 false, label %bb1, label %bb2
bb2:
  %foo = phi i32* [ null, %bb0 ], [ %zed, %bb1 ]
  %storemerge = load i32, i32* %foo
  ret i32 %storemerge
}
