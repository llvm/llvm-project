; REQUIRES: asserts
; RUN: not --crash llc -march=amdgcn -mcpu=gfx900 < %s
; RUN: not --crash llc -global-isel -march=amdgcn -mcpu=gfx900 < %s

define amdgpu_ps ptr addrspace(8) @basic_raw_buffer(ptr addrspace(3) inreg %p) {
  %rsrc = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p3(ptr addrspace(3) %p, i16 0, i32 1234, i32 5678)
  ret ptr addrspace(8) %rsrc
}
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p3(ptr addrspace(3) nocapture readnone, i16, i32, i32)
