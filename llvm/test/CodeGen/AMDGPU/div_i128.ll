; RUN: not llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs  %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: unsupported libcall legalization
define i128 @v_sdiv_i128_vv(i128 %lhs, i128 %rhs) {
  %shl = sdiv i128 %lhs, %rhs
  ret i128 %shl
}
