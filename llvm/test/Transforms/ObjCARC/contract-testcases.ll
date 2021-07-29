; RUN: opt -objc-arc-contract -S < %s | FileCheck %s
; rdar://9511608

%0 = type opaque
%1 = type opaque
%2 = type { i64, i64 }
%4 = type opaque

declare %0* @"\01-[NSAttributedString(Terminal) pathAtIndex:effectiveRange:]"(%1*, i8* nocapture, i64, %2*) optsize
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @objc_msgSend_fixup(i8*, i8*, ...)
declare i8* @objc_msgSend(i8*, i8*, ...)
declare void @llvm.objc.release(i8*)
declare %2 @NSUnionRange(i64, i64, i64, i64) optsize
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare i8* @llvm.objc.autorelease(i8*)
declare i32 @__gxx_personality_sj0(...)

; Don't get in trouble on bugpointed code.

; CHECK-LABEL: define void @test0(
define void @test0() {
bb:
  %tmp = bitcast %4* undef to i8*
  %tmp1 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %tmp) nounwind
  br label %bb3

bb3:                                              ; preds = %bb2
  br i1 undef, label %bb6, label %bb4

bb4:                                              ; preds = %bb3
  switch i64 undef, label %bb5 [
    i64 9223372036854775807, label %bb6
    i64 0, label %bb6
  ]

bb5:                                              ; preds = %bb4
  br label %bb6

bb6:                                              ; preds = %bb5, %bb4, %bb4, %bb3
  %tmp7 = phi %4* [ undef, %bb5 ], [ undef, %bb4 ], [ undef, %bb3 ], [ undef, %bb4 ]
  unreachable
}

; When rewriting operands for a phi which has multiple operands
; for the same block, use the exactly same value in each block.

; CHECK-LABEL: define void @test1(
; CHECK: %0 = bitcast i8* %tmp3 to %0* 
; CHECK: br i1 undef, label %bb7, label %bb7
; CHECK: bb7:
; CHECK: %tmp8 = phi %0* [ %0, %bb ], [ %0, %bb ]
; CHECK: }
define void @test1() {
bb:
  %tmp = tail call %0* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %0* ()*)()
  %tmp2 = bitcast %0* %tmp to i8*
  %tmp3 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %tmp2) nounwind
  br i1 undef, label %bb7, label %bb7

bb7:                                              ; preds = %bb6, %bb6, %bb5
  %tmp8 = phi %0* [ %tmp, %bb ], [ %tmp, %bb ]
  unreachable
}

; When looking for the defining instruction for an objc_retainAutoreleasedReturnValue
; call, handle the case where it's an invoke in a different basic block.
; rdar://11714057

; CHECK: define void @_Z6doTestP8NSString() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
; CHECK: invoke.cont:                                      ; preds = %entry
; CHECK-NEXT: call void asm sideeffect "mov\09r7, r7\09\09@ marker for objc_retainAutoreleaseReturnValue", ""()
; CHECK-NEXT: %tmp = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call) [[NUW:#[0-9]+]]
; CHECK: }
define void @_Z6doTestP8NSString() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  %call = invoke i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* ()*)()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %tmp = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %call) nounwind
  unreachable

lpad:                                             ; preds = %entry
  %tmp1 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov\09r7, r7\09\09@ marker for objc_retainAutoreleaseReturnValue"}

; CHECK: attributes #0 = { optsize }
; CHECK: attributes [[NUW]] = { nounwind }
