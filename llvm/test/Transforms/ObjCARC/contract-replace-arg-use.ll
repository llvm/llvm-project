; RUN: opt -objc-arc-contract -S < %s | FileCheck %s

declare i8* @objc_autoreleaseReturnValue(i8*)
declare i8* @foo1()

; Check that ARC contraction replaces the function return with the value
; returned by @objc_autoreleaseReturnValue.

; CHECK: %[[V0:[0-9]+]] = tail call i8* @objc_autoreleaseReturnValue(
; CHECK: %[[V1:[0-9]+]] = bitcast i8* %[[V0]] to i32*
; CHECK: ret i32* %[[V1]]

define i32* @autoreleaseRVTailCall() {
  %1 = call i8* @foo1()
  %2 = bitcast i8* %1 to i32*
  %3 = tail call i8* @objc_autoreleaseReturnValue(i8* %1)
  ret i32* %2
}
