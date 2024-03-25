; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX90A %s
; RUN: not --crash llc -global-isel < %s -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs 2>&1 | FileCheck %s -check-prefix=GFX908

declare float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) nocapture, float)
declare <2 x half> @llvm.amdgcn.global.atomic.fadd.f32.p1.v2f16(ptr addrspace(1) nocapture, <2 x half>)

; GFX908: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.global.atomic.fadd)

; GFX90A-LABEL: {{^}}global_atomic_fadd_f32_rtn:
; GFX90A: global_atomic_add_f32 v0, v[0:1], v2, off glc
define float @global_atomic_fadd_f32_rtn(ptr addrspace(1) %ptr, float %data) {
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) %ptr, float %data)
  ret float %ret
}

; GFX90A-LABEL: {{^}}global_atomic_fadd_v2f16_rtn:
; GFX90A: global_atomic_pk_add_f16 v0, v[0:1], v2, off glc
define <2 x half> @global_atomic_fadd_v2f16_rtn(ptr addrspace(1) %ptr, <2 x half> %data) {
  %ret = call <2 x half> @llvm.amdgcn.global.atomic.fadd.f32.p1.v2f16(ptr addrspace(1) %ptr, <2 x half> %data)
  ret <2 x half> %ret
}
