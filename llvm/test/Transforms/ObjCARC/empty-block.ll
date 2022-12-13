; RUN: opt -S -passes=objc-arc < %s | FileCheck %s
; rdar://10210274

%0 = type opaque

declare ptr @llvm.objc.retain(ptr)

declare void @llvm.objc.release(ptr)

declare ptr @llvm.objc.autoreleaseReturnValue(ptr)

; Don't delete the autorelease.

; CHECK-LABEL: define ptr @test0(
; CHECK:   @llvm.objc.retain
; CHECK: .lr.ph:
; CHECK-NOT: @llvm.objc.r
; CHECK: @llvm.objc.autoreleaseReturnValue
; CHECK-NOT: @llvm.objc.
; CHECK: }
define ptr @test0(ptr %buffer) nounwind {
  %1 = tail call ptr @llvm.objc.retain(ptr %buffer) nounwind
  br i1 undef, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph, %0
  br i1 false, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %0
  %2 = tail call ptr @llvm.objc.retain(ptr %buffer) nounwind
  tail call void @llvm.objc.release(ptr %buffer) nounwind, !clang.imprecise_release !0
  %3 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %buffer) nounwind
  ret ptr %buffer
}

; Do delete the autorelease, even with the retain in a different block.

; CHECK-LABEL: define ptr @test1(
; CHECK-NOT: @objc
; CHECK: }
define ptr @test1() nounwind {
  %buffer = call ptr @foo()
  %1 = tail call ptr @llvm.objc.retain(ptr %buffer) nounwind
  br i1 undef, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph, %0
  br i1 false, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %0
  %2 = tail call ptr @llvm.objc.retain(ptr %buffer) nounwind
  tail call void @llvm.objc.release(ptr %buffer) nounwind, !clang.imprecise_release !0
  %3 = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %buffer) nounwind
  ret ptr %buffer
}

declare ptr @foo()

!0 = !{}
