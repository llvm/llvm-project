; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
target triple = "x86_64-apple-darwin"

; In inliner, we assume that inline asm does not throw. This testing case makes
; sure that the inliner does not convert "call asm" to "invoke asm".
; rdar://15317907
; CHECK-LABEL: @caller
; Make sure we are generating "call asm" instead of "invoke asm".
; CHECK: call void asm
; CHECK-LABEL: @callee_with_asm
define void @caller() personality ptr @__objc_personality_v0 {
  br i1 undef, label %1, label %4

; <label>:1
  invoke void @callee_with_asm()
          to label %4 unwind label %2

; <label>:2
  %3 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } undef

; <label>:4
  ret void
}

define void @callee_with_asm() {
  call void asm sideeffect "mov\09r7, r7\09\09@ marker for objc_retainAutoreleaseReturnValue", ""() nounwind
  ret void
}

declare i32 @__objc_personality_v0(...)
