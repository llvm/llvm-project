; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -stop-after=amdgpu-isel -o - %s | FileCheck %s --check-prefix=SDAG

define amdgpu_kernel void @ptr_s_buffer_load_mmo(ptr addrspace(1) %out, ptr addrspace(8) inreg %rsrc, i32 inreg %offset) {
; SDAG-LABEL: name: ptr_s_buffer_load_mmo
; SDAG: S_BUFFER_LOAD_DWORD_SGPR_IMM {{.*}} :: (dereferenceable invariant load (s32) from %ir.rsrc, align 1, addrspace 8)
  %load = call i32 @llvm.amdgcn.ptr.s.buffer.load.i32(ptr addrspace(8) align 1 %rsrc, i32 %offset, i32 0), !invariant.load !0
  store i32 %load, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @ptr_s_buffer_load_mmo_overaligned(ptr addrspace(1) %out, ptr addrspace(8) inreg %rsrc, i32 inreg %offset) {
; SDAG-LABEL: name: ptr_s_buffer_load_mmo_overaligned
; SDAG: S_BUFFER_LOAD_DWORD_SGPR_IMM {{.*}} :: (dereferenceable invariant load (s32) from %ir.rsrc, align 1, addrspace 8)
  %load = call i32 @llvm.amdgcn.ptr.s.buffer.load.i32(ptr addrspace(8) align 8 %rsrc, i32 %offset, i32 0), !invariant.load !0
  store i32 %load, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.ptr.s.buffer.load.i32(ptr addrspace(8) nocapture readonly, i32, i32 immarg)

!0 = !{}
