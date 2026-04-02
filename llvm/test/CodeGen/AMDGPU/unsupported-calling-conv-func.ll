; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: unsupported calling convention
define aarch64_sve_vector_pcs void @func() {
  ret void
}
