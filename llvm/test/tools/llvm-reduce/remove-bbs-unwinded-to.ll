; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS0 --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL0 %s < %t.0

; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS1 --test-arg %s --test-arg --input-file %s -o %t.1
; RUN: FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL1 %s < %t.1

declare i32 @maybe_throwing_callee()

; CHECK-ALL: declare void @did_not_throw(i32)
declare void @did_not_throw(i32)

declare void @thrown()

; CHECK-ALL: define void @caller()
define void @caller() personality ptr @__gxx_personality_v0 {
; CHECK-ALL: bb:
; CHECK-FINAL1-NEXT: br label %bb1
bb:
; CHECK-INTERESTINGNESS0: label %bb3
; CHECK-FINAL0: br label %bb3
  %i0 = invoke i32 @maybe_throwing_callee()
          to label %bb3 unwind label %bb1

bb1:
  landingpad { ptr, i32 } catch ptr null
; CHECK-INTERESTINGNESS1: call void @thrown()

; CHECK-FINAL1: bb1:
; CHECK-FINAL1-NEXT: call void @thrown()
; CHECK-FINAL1-NEXT: ret void
  call void @thrown()
  br label %bb4

; CHECK-INTERESTINGNESS0: bb3:
; CHECK-FINAL0: bb3:
bb3:
; CHECK-INTERESTINGNESS0: call void @did_not_throw(i32
; CHECK-INTERESTINGNESS0: br label %bb4

; CHECK-FINAL0: call void @did_not_throw(i32 0)
; CHECK-FINAL0: br label %bb4

  call void @did_not_throw(i32 %i0)
  br label %bb4

; RESULT0: bb4:
; RESULT0-NEXT: ret void

; CHECK-INTERESTINGNESS0: bb4:
; CHECK-INTERESTINGNESS0-NEXT: ret void
bb4:
  ret void
}


declare i32 @__gxx_personality_v0(...)
