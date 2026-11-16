; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: SPIR-V entry point function must have a name

define spir_kernel void @0() {
  ret void
}
