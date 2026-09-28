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

declare ptr @llvm.objc.retain(ptr)
declare void @use_pointer(ptr)

; The pointer operand of a lifetime intrinsic has to be an alloca, so it must
; not be replaced by the value returned by @llvm.objc.retain.

; CHECK-LABEL: define void @lifetimeOfRetainedAlloca(
; CHECK: %[[BLOCK:.*]] = alloca ptr, align 8
; CHECK: call void @llvm.lifetime.start.p0(ptr %[[BLOCK]])
; CHECK: %[[V0:.*]] = call ptr @llvm.objc.retain(ptr %[[BLOCK]])
; CHECK: call void @use_pointer(ptr %[[V0]])
; CHECK: call void @llvm.lifetime.end.p0(ptr %[[BLOCK]])

define void @lifetimeOfRetainedAlloca() {
entry:
  %block = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(ptr %block)
  %v0 = call ptr @llvm.objc.retain(ptr %block)
  call void @use_pointer(ptr %block)
  call void @llvm.lifetime.end.p0(ptr %block)
  ret void
}
