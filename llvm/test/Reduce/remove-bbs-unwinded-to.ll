; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

declare i32 @maybe_throwing_callee()

; CHECK-ALL: declare void @did_not_throw(i32)
declare void @did_not_throw(i32)

declare void @thrown()

; CHECK-ALL: define void @caller()
define void @caller() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-ALL: bb:
bb:
; CHECK-INTERESTINGNESS: label %bb3
; CHECK-FINAL: br label %bb3
  %i0 = invoke i32 @maybe_throwing_callee()
          to label %bb3 unwind label %bb1

bb1:
  landingpad { i8*, i32 } catch i8* null
  call void @thrown()
  br label %bb4

; CHECK-ALL: bb3:
bb3:
; CHECK-INTERESTINGNESS: call void @did_not_throw(i32
; CHECK-FINAL: call void @did_not_throw(i32 undef)
; CHECK-ALL: br label %bb4
  call void @did_not_throw(i32 %i0)
  br label %bb4

; CHECK-ALL: bb4:
; CHECK-ALL: ret void
bb4:
  ret void
}

declare i32 @__gxx_personality_v0(...)
