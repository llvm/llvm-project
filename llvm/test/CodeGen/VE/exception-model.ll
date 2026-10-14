; VE defaults to SjLj exception handling.

; FIXME: The SjLj dispatch block fails the machine verifier.
; RUN: llc -mtriple=ve -verify-machineinstrs=0 < %s | FileCheck %s

; CHECK-LABEL: f:
; CHECK: _Unwind_SjLj_Register
; CHECK: _Unwind_SjLj_Unregister

declare void @g()
declare i32 @__gxx_personality_sj0(...)

define void @f() personality ptr @__gxx_personality_sj0 {
  invoke void @g() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %p = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %p
}
