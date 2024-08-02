; RUN: llc -mtriple aarch64 < %s | FileCheck %s

; The BTI instruction cannot be outlined, because it needs to be the very first
; instruction executed after an indirect call.

@g = hidden global i32 0, align 4

define hidden void @foo() minsize "branch-target-enforcement" {
entry:
; CHECK: hint #34
; CHECK: b       OUTLINED_FUNCTION_0
  store volatile i32 1, ptr @g, align 4
  ret void
}

define hidden void @bar() minsize "branch-target-enforcement" {
entry:
; CHECK: hint #34
; CHECK: b       OUTLINED_FUNCTION_0
  store volatile i32 1, ptr @g, align 4
  ret void
}
