; RUN: not llc -mtriple=aarch64-- -filetype=null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: unsupported calling convention
define amdgpu_gfx void @amdgpu_gfx_func_definition() {
  ret void
}
