; RUN: opt -S -objc-arc < %s | FileCheck %s
; rdar://10210274

%0 = type opaque

declare i8* @objc_retain(i8*)

declare void @objc_release(i8*)

declare i8* @objc_autoreleaseReturnValue(i8*)

; Don't delete the autorelease.

; CHECK: define %0* @test0(
; CHECK:   @objc_retain
; CHECK: .lr.ph:
; CHECK-NOT: @objc_r
; CHECK: @objc_autoreleaseReturnValue
; CHECK-NOT: @objc_
; CHECK: }
define %0* @test0(%0* %buffer) nounwind {
  %1 = bitcast %0* %buffer to i8*
  %2 = tail call i8* @objc_retain(i8* %1) nounwind
  br i1 undef, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph, %0
  br i1 false, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %0
  %3 = tail call i8* @objc_retain(i8* %1) nounwind
  tail call void @objc_release(i8* %1) nounwind, !clang.imprecise_release !0
  %4 = tail call i8* @objc_autoreleaseReturnValue(i8* %1) nounwind
  ret %0* %buffer
}

; Do delete the autorelease, even with the retain in a different block.

; CHECK: define %0* @test1(
; CHECK-NOT: @objc
; CHECK: }
define %0* @test1() nounwind {
  %buffer = call %0* @foo()
  %1 = bitcast %0* %buffer to i8*
  %2 = tail call i8* @objc_retain(i8* %1) nounwind
  br i1 undef, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph, %0
  br i1 false, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %0
  %3 = tail call i8* @objc_retain(i8* %1) nounwind
  tail call void @objc_release(i8* %1) nounwind, !clang.imprecise_release !0
  %4 = tail call i8* @objc_autoreleaseReturnValue(i8* %1) nounwind
  ret %0* %buffer
}

declare %0* @foo()

!0 = metadata !{}
