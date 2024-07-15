; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx90a -verify-machineinstrs | FileCheck %s -check-prefix=GFX90A

declare float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1), float)
declare <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1), <2 x half>)

; GFX90A-LABEL: {{^}}global_atomic_add_f32:
; GFX90A: global_atomic_add_f32 v0, v[0:1], v2, off glc
define amdgpu_ps float @global_atomic_add_f32(ptr addrspace(1) %ptr, float %data) {
main_body:
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) %ptr, float %data)
  ret float %ret
}

; GFX90A-LABEL: {{^}}global_atomic_add_f32_off4:
; GFX90A: global_atomic_add_f32 v0, v[0:1], v2, off offset:4 glc
define amdgpu_ps float @global_atomic_add_f32_off4(ptr addrspace(1) %ptr, float %data) {
main_body:
  %p = getelementptr float, ptr addrspace(1) %ptr, i64 1
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) %p, float %data)
  ret float %ret
}

; GFX90A-LABEL: {{^}}global_atomic_add_f32_offneg4:
; GFX90A: global_atomic_add_f32 v0, v[0:1], v2, off offset:-4 glc
define amdgpu_ps float @global_atomic_add_f32_offneg4(ptr addrspace(1) %ptr, float %data) {
main_body:
  %p = getelementptr float, ptr addrspace(1) %ptr, i64 -1
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) %p, float %data)
  ret float %ret
}

; GFX90A-LABEL: {{^}}global_atomic_pk_add_v2f16:
; GFX90A: global_atomic_pk_add_f16 v0, v[0:1], v2, off glc
define amdgpu_ps <2 x half> @global_atomic_pk_add_v2f16(ptr addrspace(1) %ptr, <2 x half> %data) {
main_body:
  %ret = call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1) %ptr, <2 x half> %data)
  ret <2 x half> %ret
}

; GFX90A-LABEL: {{^}}global_atomic_pk_add_v2f16_off4:
; GFX90A: global_atomic_pk_add_f16 v0, v[0:1], v2, off offset:4 glc
define amdgpu_ps <2 x half> @global_atomic_pk_add_v2f16_off4(ptr addrspace(1) %ptr, <2 x half> %data) {
main_body:
  %p = getelementptr <2 x half>, ptr addrspace(1) %ptr, i64 1
  %ret = call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1) %p, <2 x half> %data)
  ret <2 x half> %ret
}

; GFX90A-LABEL: {{^}}global_atomic_pk_add_v2f16_offneg4:
; GFX90A: global_atomic_pk_add_f16 v0, v[0:1], v2, off offset:-4 glc
define amdgpu_ps <2 x half> @global_atomic_pk_add_v2f16_offneg4(ptr addrspace(1) %ptr, <2 x half> %data) {
main_body:
  %p = getelementptr <2 x half>, ptr addrspace(1) %ptr, i64 -1
  %ret = call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1) %p, <2 x half> %data)
  ret <2 x half> %ret
}
