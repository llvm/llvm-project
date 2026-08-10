; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare ptr @foo1()

; Check that ARC contraction replaces the function return with the value
; returned by @llvm.objc.autoreleaseReturnValue.

; CHECK-LABEL: define ptr @autoreleaseRVTailCall(
; CHECK: %[[V0:[0-9]+]] = tail call ptr @llvm.objc.autoreleaseReturnValue(
; CHECK: ret ptr %[[V0]]

define ptr @autoreleaseRVTailCall() {
  %1 = call ptr @foo1()
  %2 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %1)
  ret ptr %1
}

declare ptr @foo2(i32);

; CHECK-LABEL: define ptr @autoreleaseRVTailCallPhi(
; CHECK: %[[PHIVAL:.*]] = phi ptr [ %{{.*}}, %bb1 ], [ %{{.*}}, %bb2 ]
; CHECK: %[[RETVAL:.*]] = phi ptr [ %{{.*}}, %bb1 ], [ %{{.*}}, %bb2 ]
; CHECK: %[[V4:.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %[[PHIVAL]])
; CHECK: ret ptr %[[V4]]

define ptr @autoreleaseRVTailCallPhi(i1 %cond) {
entry:
  br i1 %cond, label %bb1, label %bb2
bb1:
  %v0 = call ptr @foo2(i32 1)
  br label %bb3
bb2:
  %v2 = call ptr @foo2(i32 2)
  br label %bb3
bb3:
  %phival = phi ptr [ %v0, %bb1 ], [ %v2, %bb2 ]
  %retval = phi ptr [ %v0, %bb1 ], [ %v2, %bb2 ]
  %v4 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %phival)
  ret ptr %retval
}
