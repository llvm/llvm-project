; FIXME: This should error:
; RUN: llc -mtriple=aarch64-- -filetype=null %s
declare amdgpu_gfx void @amdgpu_gfx_func()

define void @call_amdgpu_gfx_func() {
  call amdgpu_gfx void @amdgpu_gfx_func()
  ret void
}
