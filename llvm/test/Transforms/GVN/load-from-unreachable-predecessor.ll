; RUN: opt -passes=gvn -S < %s | FileCheck %s

; Check that an unreachable predecessor to a PHI node doesn't cause a crash.
; PR21625.

define i32 @f(ptr %f) {
; CHECK: bb0:
; Load should be removed, since it's ignored.
; CHECK-NEXT: br label
bb0:
  %bar = load ptr, ptr %f
  br label %bb2
bb1:
  %zed = load ptr, ptr %f
  br i1 false, label %bb1, label %bb2
bb2:
  %foo = phi ptr [ null, %bb0 ], [ %zed, %bb1 ]
  %storemerge = load i32, ptr %foo
  ret i32 %storemerge
}
