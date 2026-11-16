; RUN: not llc -mtriple=aarch64-- -global-isel=0 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=aarch64-- -global-isel=1 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: unsupported calling convention
declare amdgpu_gfx void @amdgpu_gfx_func()

define void @call_amdgpu_gfx_func() {
  call amdgpu_gfx void @amdgpu_gfx_func()
  ret void
}
