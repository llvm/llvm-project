; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

; Test that we don't break the callsite for an unhandled invoke

declare void @did_not_throw(i32)
declare void @thrown()

; INTERESTING-LABEL: define i32 @maybe_throwing_callee(

; REDUCED-LABEL: define i32 @maybe_throwing_callee(i32 %arg0, i32 %arg1) {
define i32 @maybe_throwing_callee(i32 %arg0, i32 %arg1) {
  call void @thrown()
  ret i32 %arg1
}

@initializer_user = global [1 x ptr] [ptr @maybe_throwing_callee ]

; REDUCED-LABEL: define void @caller()
; REDUCED: %i0 = invoke i32 @maybe_throwing_callee(i32 0, i32 0) #0
define void @caller(i32 %arg0, ptr %arg1) personality ptr @__gxx_personality_v0 {
bb:
  %val = load i32, ptr %arg1
  %i0 = invoke i32 @maybe_throwing_callee(i32 0, i32 0) nofree
          to label %bb3 unwind label %bb1

bb1:
  landingpad { ptr, i32 }
  catch ptr null
  call void @thrown()
  br label %bb4

bb3:
  call void @did_not_throw(i32 %i0)
  br label %bb4

bb4:
  ret void
}

declare i32 @__gxx_personality_v0(...)
