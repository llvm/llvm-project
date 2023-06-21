; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX90A %s
; RUN: not --crash llc -global-isel < %s -march=amdgcn -mcpu=gfx908 -verify-machineinstrs 2>&1 | FileCheck %s -check-prefix=GFX908

declare float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(float, ptr addrspace(8), i32, i32, i32 immarg)
declare <2 x half> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2f16(<2 x half>, ptr addrspace(8), i32, i32, i32 immarg)

; GFX908: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_AMDGPU_BUFFER_ATOMIC_FADD %{{[0-9]+}}:vgpr, %{{[0-9]+}}:sgpr(<4 x s32>), %{{[0-9]+}}:vgpr(s32), %{{[0-9]+}}:vgpr, %{{[0-9]+}}:sgpr, 0, 0, 0 :: (volatile dereferenceable load store (s32) on %ir.rsrc.load, align 1, addrspace 8) (in function: buffer_atomic_add_f32_rtn)

; GFX90A-LABEL: {{^}}buffer_atomic_add_f32_rtn:
; GFX90A: buffer_atomic_add_f32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}], s{{[0-9]+}} offen glc
define amdgpu_kernel void @buffer_atomic_add_f32_rtn(float %val, ptr addrspace(8) inreg %rsrc, i32 %voffset, i32 %soffset) {
main_body:
  %ret = call float @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.f32(float %val, ptr addrspace(8) %rsrc, i32 %voffset, i32 %soffset, i32 0)
  store float %ret, ptr undef
  ret void
}

; GFX90A-LABEL: {{^}}buffer_atomic_add_v2f16_rtn:
; GFX90A: buffer_atomic_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}], s{{[0-9]+}} offen glc
define amdgpu_kernel void @buffer_atomic_add_v2f16_rtn(<2 x half> %val, ptr addrspace(8) inreg %rsrc, i32 %voffset, i32 inreg %soffset) {
main_body:
  %ret = call <2 x half> @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.v2f16(<2 x half> %val, ptr addrspace(8) %rsrc, i32 %voffset, i32 %soffset, i32 0)
  store <2 x half> %ret, ptr undef
  ret void
}
