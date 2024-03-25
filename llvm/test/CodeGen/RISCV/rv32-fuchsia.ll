; RUN: not --crash llc -mtriple=riscv32-unknown-fuchsia < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Fuchsia is only supported for 64-bit
define void @nothing() nounwind {
  ret void
}
