; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx90a -verify-machineinstrs | FileCheck %s -check-prefix=GFX90A
; RUN: not --crash llc < %s -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs 2>&1 | FileCheck %s -check-prefix=GFX908

declare float @llvm.amdgcn.buffer.atomic.fadd.f32(float, <4 x i32>, i32, i32, i1)
declare <2 x half> @llvm.amdgcn.buffer.atomic.fadd.v2f16(<2 x half>, <4 x i32>, i32, i32, i1)
declare float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1), float)
declare <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1), <2 x half>)

; GFX908:  LLVM ERROR: Cannot select: {{.+}}: f32,ch = BUFFER_ATOMIC_FADD

; GFX90A-LABEL: {{^}}buffer_atomic_add_f32:
; GFX90A: buffer_atomic_add_f32 v0, v1, s[0:3], 0 idxen glc
define amdgpu_ps float @buffer_atomic_add_f32(<4 x i32> inreg %rsrc, float %data, i32 %vindex) {
main_body:
  %ret = call float @llvm.amdgcn.buffer.atomic.fadd.f32(float %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  ret float %ret
}

; GFX90A-LABEL: {{^}}buffer_atomic_add_f32_off4_slc:
; GFX90A: buffer_atomic_add_f32 v0, v1, s[0:3], 0 idxen offset:4 glc slc
define amdgpu_ps float @buffer_atomic_add_f32_off4_slc(<4 x i32> inreg %rsrc, float %data, i32 %vindex) {
main_body:
  %ret = call float @llvm.amdgcn.buffer.atomic.fadd.f32(float %data, <4 x i32> %rsrc, i32 %vindex, i32 4, i1 1)
  ret float %ret
}

; GFX90A-LABEL: {{^}}buffer_atomic_pk_add_v2f16:
; GFX90A: buffer_atomic_pk_add_f16 v0, v1, s[0:3], 0 idxen glc
define amdgpu_ps <2 x half> @buffer_atomic_pk_add_v2f16(<4 x i32> inreg %rsrc, <2 x half> %data, i32 %vindex) {
main_body:
  %ret = call <2 x half> @llvm.amdgcn.buffer.atomic.fadd.v2f16(<2 x half> %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  ret <2 x half> %ret
}

; GFX90A-LABEL: {{^}}buffer_atomic_pk_add_v2f16_off4_slc:
; GFX90A: buffer_atomic_pk_add_f16 v0, v1, s[0:3], 0 idxen offset:4 glc slc
define amdgpu_ps <2 x half> @buffer_atomic_pk_add_v2f16_off4_slc(<4 x i32> inreg %rsrc, <2 x half> %data, i32 %vindex) {
main_body:
  %ret = call <2 x half> @llvm.amdgcn.buffer.atomic.fadd.v2f16(<2 x half> %data, <4 x i32> %rsrc, i32 %vindex, i32 4, i1 1)
  ret <2 x half> %ret
}

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
