; RUN: not llc -mtriple=amdgpu9.00-amd-amdhsa -filetype=null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: unsupported calling convention
define void @caller(ptr %func) {
  call aarch64_sve_vector_pcs void %func()
  ret void
}
