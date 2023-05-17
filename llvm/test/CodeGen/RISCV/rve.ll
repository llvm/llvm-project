; RUN: not --crash llc -mtriple=riscv32 -mattr=+e < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=riscv64 -mattr=+e < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Codegen not yet implemented for RVE

define void @nothing() nounwind {
  ret void
}
