; AVR has no exception handling, so invokes are lowered to calls.

; RUN: llc -mtriple=avr < %s | FileCheck %s

; CHECK-LABEL: f:
; CHECK: call g
; CHECK-NOT: _Unwind

declare void @g() addrspace(1)
declare i32 @__gxx_personality_v0(...) addrspace(1)

define void @f() addrspace(1) personality ptr addrspace(1) @__gxx_personality_v0 {
  invoke void @g() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %p = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %p
}
