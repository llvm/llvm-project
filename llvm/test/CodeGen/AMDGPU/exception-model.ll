; AMDGPU supports no exception model, so the invoke is lowered to a call and
; even a "dwarf" model emits no EH tables or CFI.

; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck %s

; CHECK-LABEL: f:
; CHECK-NOT: .cfi_startproc
; CHECK-NOT: .cfi_lsda
; CHECK-NOT: gcc_except_table
; CHECK-NOT: _Unwind

declare void @g()
declare i32 @__gxx_personality_v0(...)

define void @f() personality ptr @__gxx_personality_v0 {
  invoke void @g() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %p = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %p
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"dwarf"}
