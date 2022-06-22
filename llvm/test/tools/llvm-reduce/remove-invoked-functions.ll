; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL %s

; CHECK-INTERESTINGNESS: define i32 @maybe_throwing_callee(
; CHECK-FINAL: define i32 @maybe_throwing_callee()
define i32 @maybe_throwing_callee(i32 %arg) {
; CHECK-ALL: call void @thrown()
; CHECK-INTERESTINGNESS: ret i32
; CHECK-FINAL: ret i32 0
  call void @thrown()
  ret i32 %arg
}

; CHECK-ALL: declare void @did_not_throw(i32)
declare void @did_not_throw(i32)

; CHECK-ALL: declare void @thrown()
declare void @thrown()

; CHECK-INTERESTINGNESS: define void @caller(
; CHECK-FINAL: define void @caller()
define void @caller(i32 %arg) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-ALL: bb:
bb:
; CHECK-INTERESTINGNESS: %i0 = invoke i32 {{.*}}@maybe_throwing_callee
; CHECK-FINAL: %i0 = invoke i32 bitcast (i32 ()* @maybe_throwing_callee to i32 (i32)*)
; CHECK-ALL: to label %bb3 unwind label %bb1
  %i0 = invoke i32 @maybe_throwing_callee(i32 %arg)
          to label %bb3 unwind label %bb1

; CHECK-ALL: bb1:
bb1:
; CHECK-ALL: landingpad { i8*, i32 }
; CHECK-ALL: catch i8* null
; CHECK-ALL: call void @thrown()
; CHECK-ALL: br label %bb4
  landingpad { i8*, i32 }
  catch i8* null
  call void @thrown()
  br label %bb4

; CHECK-ALL: bb3:
bb3:
; CHECK-ALL: call void @did_not_throw(i32 %i0)
; CHECK-ALL: br label %bb4
  call void @did_not_throw(i32 %i0)
  br label %bb4

; CHECK-ALL: bb4:
; CHECK-ALL: ret void
bb4:
  ret void
}

declare i32 @__gxx_personality_v0(...)
