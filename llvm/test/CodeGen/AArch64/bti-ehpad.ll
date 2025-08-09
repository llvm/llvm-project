; llvm/test/CodeGen/AArch64/bti-ehpad.ll
; REQUIRES: aarch64-registered-target
; RUN: llc -mtriple=aarch64-none-linux-gnu %s -o - | FileCheck %s

declare i32 @__gxx_personality_v0(...)

define void @test() #0 personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw()
          to label %ret unwind label %lpad
lpad:                               ; catch.dispatch
  landingpad { ptr, i32 }
          cleanup
  ret void
ret:
  ret void
}

declare void @may_throw()

attributes #0 = { "branch-target-enforcement"="true" }

; Function needs both the architectural feature *and* the enforcement request.
attributes #0 = { "branch-target-enforcement"="true" "target-features"="+bti" }

; CHECK:      bti
