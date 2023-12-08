; Test the invoke reduction standalone, dead blocks are not removed
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=invokes --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --check-prefixes=CHECK,RESULT,RESULT-SINGLE %s < %t.0

; Test the full reduction pipeline, which cleans up unreachable blocks
; RUN: llvm-reduce --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t.1
; RUN: FileCheck --check-prefixes=CHECK,RESULT,RESULT-FULL %s < %t.1


define i32 @maybe_throwing_callee(i32 %arg) {
  call void @thrown()
  ret i32 %arg
}

declare void @did_not_throw(i32)

declare void @thrown()

; CHECK-LABEL: define void @invoke_keep_landingpad_block(i32 %arg) personality ptr @__gxx_personality_v0 {
; INTERESTING: i32 @maybe_throwing_callee(

; RESULT-SINGLE: %i0 = call i32 @maybe_throwing_callee(i32 %arg), !some.metadata !0
; RESULT-SINGLE-NEXT: br label %bb3

; RESULT-FULL: %i0 = call i32 @maybe_throwing_callee()
; RESULT-FULL-NEXT: br label %bb4


; RESULT-SINGLE: bb1:  ; No predecessors!
; RESULT-SINGLE-NEXT: %landing = landingpad { ptr, i32 }
; RESULT-SINGLE-NEXT: catch ptr null
; RESULT-SINGLE-NEXT: call void @thrown()
define void @invoke_keep_landingpad_block(i32 %arg) personality ptr @__gxx_personality_v0 {
bb:
  %i0 = invoke i32 @maybe_throwing_callee(i32 %arg)
          to label %bb3 unwind label %bb1, !some.metadata !0

bb1:                                              ; preds = %bb
  %landing = landingpad { ptr, i32 }
          catch ptr null
  ; INTERESTING: call void @thrown()
  call void @thrown()
  br label %bb4

bb3:                                              ; preds = %bb
  call void @did_not_throw(i32 %i0)
  br label %bb4

bb4:                                              ; preds = %bb3, %bb1
  ret void
}

; CHECK-LABEL: define void @invoke_drop_landingpad_block(i32 %arg) personality ptr @__gxx_personality_v0 {
; INTERESTING: i32 @maybe_throwing_callee(

; RESULT-SINGLE: %i0 = call i32 @maybe_throwing_callee(i32 %arg), !some.metadata !0
; RESULT-SINGLE-NEXT: br label %bb3

; RESULT-SINGLE: bb1:  ; No predecessors!
; RESULT-SINGLE-NEXT: %landing = landingpad { ptr, i32 }

; RESULT-SINGLE: bb3:
; RESULT-SINGLE-NEXT: call void @did_not_throw(i32 %i0)

; RESULT-FULL: %i0 = call i32 @maybe_throwing_callee()
; RESULT-FULL-NEXT: call void @did_not_throw()
; RESULT-FULL-NEXT: ret void
define void @invoke_drop_landingpad_block(i32 %arg) personality ptr @__gxx_personality_v0 {
bb:
  %i0 = invoke i32 @maybe_throwing_callee(i32 %arg)
          to label %bb3 unwind label %bb1, !some.metadata !0

bb1:                                              ; preds = %bb
  %landing = landingpad { ptr, i32 }
          catch ptr null
  call void @thrown()
  br label %bb4

bb3:                                              ; preds = %bb
  ; INTERESTING: call void @did_not_throw(
  call void @did_not_throw(i32 %i0)
  br label %bb4

bb4:                                              ; preds = %bb3, %bb1
  ret void
}

declare i32 @another_maybe_throwing_callee(i32 %arg)

; Test the same landing pad block is used by multiple invokes
; CHECK-LABEL: define i32 @multi_invoke_caller(i32 %arg) personality ptr @__gxx_personality_v0 {
define i32 @multi_invoke_caller(i32 %arg) personality ptr @__gxx_personality_v0 {
bb:
  %i0 = invoke i32 @maybe_throwing_callee(i32 %arg)
          to label %bb3 unwind label %bb1, !some.metadata !0

; RESULT: bb1:                                              ; preds = %bb4
; RESULT-NEXT: %landing = landingpad { ptr, i32 }
; RESULT-NEXT:   catch ptr null
bb1:                                              ; preds = %bb
  %landing = landingpad { ptr, i32 }
          catch ptr null
  ; INTERESTING: call void @thrown()
  call void @thrown()
  br label %bb4

bb3:                                              ; preds = %bb
  call void @did_not_throw(i32 %i0)
  br label %bb4

bb4:                                              ; preds = %bb3, %bb1
  ; INTERESTING: invoke i32 @another_maybe_throwing_callee(

  ; RESULT-SINGLE:   %i1 = invoke i32 @another_maybe_throwing_callee(i32 %arg)
  ; RESULT-SINGLE-NEXT: to label %bb5 unwind label %bb1, !some.metadata !1

  ; RESULT-FULL:   %i1 = invoke i32 @another_maybe_throwing_callee(i32 0)
  ; RESULT-FULL-NEXT: to label %bb5 unwind label %bb1{{$}}
  %i1 = invoke i32 @another_maybe_throwing_callee(i32 %arg)
          to label %bb5 unwind label %bb1, !some.metadata !1

bb5:
  ret i32 %i1
}

declare i32 @__gxx_personality_v0(...)

!0 = !{!"arst"}
!1 = !{!"arstarst"}
