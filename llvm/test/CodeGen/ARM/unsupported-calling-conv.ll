; RUN: not --crash llc -mtriple=armv7-none-eabi < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Unsupported calling convention
define coldcc void @f() {
  ret void
}
