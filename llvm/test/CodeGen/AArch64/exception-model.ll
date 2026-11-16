; RUN: llc -O0 -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; CHECK: .cfi_lsda
; CHECK: GCC_except_table

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
