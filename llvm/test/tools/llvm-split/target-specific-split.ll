; REQUIRES: x86-registered-target

; RUN: llvm-split -o %t %s -mtriple x86_64 -preserve-locals 2>&1 | FileCheck %s

; Basic test for a target that doesn't support target-specific module splitting.

; CHECK: warning: --preserve-locals has no effect when using TargetMachine::splitModule
; CHECK: warning: TargetMachine::splitModule failed, falling back to default splitModule implementation

define void @bar() {
  ret void
}
