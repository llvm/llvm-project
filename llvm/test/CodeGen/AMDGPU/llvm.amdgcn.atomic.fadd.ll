; RUN: llc < %s -march=amdgcn -mcpu=gfx908 -verify-machineinstrs | FileCheck %s -check-prefix=GCN

declare float @llvm.amdgcn.buffer.atomic.fadd.f32(float, <4 x i32>, i32, i32, i1)
declare <2 x half> @llvm.amdgcn.buffer.atomic.fadd.v2f16(<2 x half>, <4 x i32>, i32, i32, i1)
declare float @llvm.amdgcn.global.atomic.fadd.f32.p1f32.f32(float addrspace(1)*, float)
declare <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1v2f16.v2f16(<2 x half> addrspace(1)*, <2 x half>)

; GCN-LABEL: {{^}}buffer_atomic_add_f32:
; GCN: buffer_atomic_add_f32 v0, v1, s[0:3], 0 idxen
define amdgpu_ps void @buffer_atomic_add_f32(<4 x i32> inreg %rsrc, float %data, i32 %vindex) {
main_body:
  %ret = call float @llvm.amdgcn.buffer.atomic.fadd.f32(float %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_atomic_add_f32_off4_slc:
; GCN: buffer_atomic_add_f32 v0, v1, s[0:3], 0 idxen offset:4 slc
define amdgpu_ps void @buffer_atomic_add_f32_off4_slc(<4 x i32> inreg %rsrc, float %data, i32 %vindex) {
main_body:
  %ret = call float @llvm.amdgcn.buffer.atomic.fadd.f32(float %data, <4 x i32> %rsrc, i32 %vindex, i32 4, i1 1)
  ret void
}

; GCN-LABEL: {{^}}buffer_atomic_pk_add_v2f16:
; GCN: buffer_atomic_pk_add_f16 v0, v1, s[0:3], 0 idxen
define amdgpu_ps void @buffer_atomic_pk_add_v2f16(<4 x i32> inreg %rsrc, <2 x half> %data, i32 %vindex) {
main_body:
  %ret = call <2 x half> @llvm.amdgcn.buffer.atomic.fadd.v2f16(<2 x half> %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_atomic_pk_add_v2f16_off4_slc:
; GCN: buffer_atomic_pk_add_f16 v0, v1, s[0:3], 0 idxen offset:4 slc
define amdgpu_ps void @buffer_atomic_pk_add_v2f16_off4_slc(<4 x i32> inreg %rsrc, <2 x half> %data, i32 %vindex) {
main_body:
  %ret = call <2 x half> @llvm.amdgcn.buffer.atomic.fadd.v2f16(<2 x half> %data, <4 x i32> %rsrc, i32 %vindex, i32 4, i1 1)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_add_f32:
; GCN: global_atomic_add_f32 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @global_atomic_add_f32(float addrspace(1)* %ptr, float %data) {
main_body:
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1f32.f32(float addrspace(1)* %ptr, float %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_add_f32_off4:
; GCN: global_atomic_add_f32 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4
define amdgpu_kernel void @global_atomic_add_f32_off4(float addrspace(1)* %ptr, float %data) {
main_body:
  %p = getelementptr float, float addrspace(1)* %ptr, i64 1
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1f32.f32(float addrspace(1)* %p, float %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_add_f32_offneg4:
; GCN: global_atomic_add_f32 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:-4
define amdgpu_kernel void @global_atomic_add_f32_offneg4(float addrspace(1)* %ptr, float %data) {
main_body:
  %p = getelementptr float, float addrspace(1)* %ptr, i64 -1
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1f32.f32(float addrspace(1)* %p, float %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_pk_add_v2f16:
; GCN: global_atomic_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @global_atomic_pk_add_v2f16(<2 x half> addrspace(1)* %ptr, <2 x half> %data) {
main_body:
  %ret = call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1v2f16.v2f16(<2 x half> addrspace(1)* %ptr, <2 x half> %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_pk_add_v2f16_off4:
; GCN: global_atomic_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4
define amdgpu_kernel void @global_atomic_pk_add_v2f16_off4(<2 x half> addrspace(1)* %ptr, <2 x half> %data) {
main_body:
  %p = getelementptr <2 x half>, <2 x half> addrspace(1)* %ptr, i64 1
  %ret = call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1v2f16.v2f16(<2 x half> addrspace(1)* %p, <2 x half> %data)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_pk_add_v2f16_offneg4:
; GCN: global_atomic_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:-4{{$}}
define amdgpu_kernel void @global_atomic_pk_add_v2f16_offneg4(<2 x half> addrspace(1)* %ptr, <2 x half> %data) {
main_body:
  %p = getelementptr <2 x half>, <2 x half> addrspace(1)* %ptr, i64 -1
  %ret = call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1v2f16.v2f16(<2 x half> addrspace(1)* %p, <2 x half> %data)
  ret void
}

; Make sure this artificially selects with an incorrect subtarget, but
; the feature set.
; GCN-LABEL: {{^}}global_atomic_fadd_f32_wrong_subtarget:
; GCN: global_atomic_add_f32 v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @global_atomic_fadd_f32_wrong_subtarget(float addrspace(1)* %ptr, float %data) #0 {
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1f32.f32(float addrspace(1)* %ptr, float %data)
  ret void
}

attributes #0 = { "target-cpu"="gfx803" "target-features"="+atomic-fadd-insts" }
