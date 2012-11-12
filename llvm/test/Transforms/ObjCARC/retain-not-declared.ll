; RUN: opt -S -objc-arc -objc-arc-contract < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
declare i8* @objc_unretainedObject(i8*)
declare i8* @objc_retainAutoreleasedReturnValue(i8*)
declare i8* @objc_autoreleaseReturnValue(i8*)
declare i8* @objc_msgSend(i8*, i8*, ...)
declare void @objc_release(i8*)

; Test that the optimizer can create an objc_retainAutoreleaseReturnValue
; declaration even if no objc_retain declaration exists.
; rdar://9401303

; CHECK:      define i8* @test0(i8* %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call i8* @objc_retainAutoreleaseReturnValue(i8* %p) nounwind
; CHECK-NEXT:   ret i8* %0
; CHECK-NEXT: }

define i8* @test0(i8* %p) {
entry:
  %call = tail call i8* @objc_unretainedObject(i8* %p)
  %0 = tail call i8* @objc_retainAutoreleasedReturnValue(i8* %call) nounwind
  %1 = tail call i8* @objc_autoreleaseReturnValue(i8* %0) nounwind
  ret i8* %1
}

; Properly create the @objc_retain declaration when it doesn't already exist.
; rdar://9825114

; CHECK: @test1(
; CHECK: @objc_retain(
; CHECK: @objc_retainAutoreleasedReturnValue(
; CHECK: @objc_release(
; CHECK: @objc_release(
; CHECK: }
define void @test1(i8* %call88) nounwind {
entry:
  %tmp1 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call88) nounwind
  %call94 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*)*)(i8* %tmp1)
          to label %invoke.cont93 unwind label %lpad91

invoke.cont93:                                    ; preds = %entry
  %tmp2 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call94) nounwind
  call void @objc_release(i8* %tmp1) nounwind
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*)*)(i8* %tmp2)
          to label %invoke.cont102 unwind label %lpad100

invoke.cont102:                                   ; preds = %invoke.cont93
  call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  unreachable

lpad91:                                           ; preds = %entry
  %exn91 = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
              cleanup
  unreachable

lpad100:                                          ; preds = %invoke.cont93
  %exn100 = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
              cleanup
  call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  unreachable
}

declare i32 @__gxx_personality_v0(...)

!0 = metadata !{}
