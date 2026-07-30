; RUN: llc <%s -mtriple=aarch64 2>&1  | FileCheck %s

; CHECK: warning: inline asm clobber list contains reserved registers: SP

define void @foo() nounwind {
  call void asm sideeffect "mov x7, #1", "~{x7},~{sp}"()
  ret void
}
