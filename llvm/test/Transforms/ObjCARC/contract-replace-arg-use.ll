; RUN: opt -objc-arc-contract -S < %s | FileCheck %s

declare i8* @objc_autoreleaseReturnValue(i8*)
declare i8* @foo1()

; Check that ARC contraction replaces the function return with the value
; returned by @objc_autoreleaseReturnValue.

; CHECK-LABEL: define i32* @autoreleaseRVTailCall(
; CHECK: %[[V0:[0-9]+]] = tail call i8* @objc_autoreleaseReturnValue(
; CHECK: %[[V1:[0-9]+]] = bitcast i8* %[[V0]] to i32*
; CHECK: ret i32* %[[V1]]

define i32* @autoreleaseRVTailCall() {
  %1 = call i8* @foo1()
  %2 = bitcast i8* %1 to i32*
  %3 = tail call i8* @objc_autoreleaseReturnValue(i8* %1)
  ret i32* %2
}

declare i32* @foo2(i32);

; CHECK-LABEL: define i32* @autoreleaseRVTailCallPhi(
; CHECK: %[[PHIVAL:.*]] = phi i8* [ %{{.*}}, %bb1 ], [ %{{.*}}, %bb2 ]
; CHECK: %[[RETVAL:.*]] = phi i32* [ %{{.*}}, %bb1 ], [ %{{.*}}, %bb2 ]
; CHECK: %[[V4:.*]] = tail call i8* @objc_autoreleaseReturnValue(i8* %[[PHIVAL]])
; CHECK: %[[V0:.*]] = bitcast i8* %[[V4]] to i32*
; CHECK: ret i32* %[[V0]]

define i32* @autoreleaseRVTailCallPhi(i1 %cond) {
entry:
  br i1 %cond, label %bb1, label %bb2
bb1:
  %v0 = call i32* @foo2(i32 1)
  %v1 = bitcast i32* %v0 to i8*
  br label %bb3
bb2:
  %v2 = call i32* @foo2(i32 2)
  %v3 = bitcast i32* %v2 to i8*
  br label %bb3
bb3:
  %phival = phi i8* [ %v1, %bb1 ], [ %v3, %bb2 ]
  %retval = phi i32* [ %v0, %bb1 ], [ %v2, %bb2 ]
  %v4 = tail call i8* @objc_autoreleaseReturnValue(i8* %phival)
  ret i32* %retval
}
