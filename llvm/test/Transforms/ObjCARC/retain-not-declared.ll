; RUN: opt -S -passes=objc-arc,objc-arc-contract < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
declare i8* @llvm.objc.unretainedObject(i8*)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare i8* @objc_msgSend(i8*, i8*, ...)
declare void @llvm.objc.release(i8*)

; Test that the optimizer can create an objc_retainAutoreleaseReturnValue
; declaration even if no objc_retain declaration exists.
; rdar://9401303

; CHECK:      define i8* @test0(i8* %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call i8* @llvm.objc.retainAutoreleaseReturnValue(i8* %p) [[NUW:#[0-9]+]]
; CHECK-NEXT:   ret i8* %0
; CHECK-NEXT: }

define i8* @test0(i8* %p) {
entry:
  %call = tail call i8* @llvm.objc.unretainedObject(i8* %p)
  %0 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call) nounwind
  %1 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call) nounwind
  ret i8* %call
}

; Properly create the @llvm.objc.retain declaration when it doesn't already exist.
; rdar://9825114

; CHECK-LABEL: @test1(
; CHECK: @llvm.objc.retain
; CHECK: @llvm.objc.retainAutoreleasedReturnValue(
; CHECK: @llvm.objc.release
; CHECK: @llvm.objc.release
; CHECK: }
define void @test1(i8* %call88) nounwind personality i32 (...)* @__gxx_personality_v0 {
entry:
  %tmp1 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call88) nounwind
  %call94 = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*)*)(i8* %tmp1)
          to label %invoke.cont93 unwind label %lpad91

invoke.cont93:                                    ; preds = %entry
  %tmp2 = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call94) nounwind
  call void @llvm.objc.release(i8* %tmp1) nounwind
  invoke void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*)*)(i8* %tmp2)
          to label %invoke.cont102 unwind label %lpad100

invoke.cont102:                                   ; preds = %invoke.cont93
  call void @llvm.objc.release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  unreachable

lpad91:                                           ; preds = %entry
  %exn91 = landingpad {i8*, i32}
              cleanup
  unreachable

lpad100:                                          ; preds = %invoke.cont93
  %exn100 = landingpad {i8*, i32}
              cleanup
  call void @llvm.objc.release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  unreachable
}

declare i32 @__gxx_personality_v0(...)

!0 = !{}

; CHECK: attributes [[NUW]] = { nounwind }
