; RUN: llc -mtriple=r600 -mcpu=redwood -filetype=null %s

define amdgpu_kernel void @f() {
  ret void
}
