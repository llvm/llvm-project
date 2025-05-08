; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SDAG %s
; xUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: GlobalISel does not work with bf16

declare float @llvm.amdgcn.tanh.f32(float) #0
declare half @llvm.amdgcn.tanh.f16(half) #0
declare bfloat @llvm.amdgcn.tanh.bf16(bfloat) #0

; GCN-LABEL: {{^}}tanh_f32:
; GCN: v_tanh_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @tanh_f32(ptr addrspace(1) %out, float %src) #1 {
  %tanh = call float @llvm.amdgcn.tanh.f32(float %src) #0
  store float %tanh, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: Really these should be constant folded
; GCN-LABEL: {{^}}tanh_f32_constant_4.0
; GCN: v_tanh_f32_e32 {{v[0-9]+}}, 4.0
define amdgpu_kernel void @tanh_f32_constant_4.0(ptr addrspace(1) %out) #1 {
  %tanh = call float @llvm.amdgcn.tanh.f32(float 4.0) #0
  store float %tanh, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}tanh_f32_constant_100.0
; GCN: v_tanh_f32_e32 {{v[0-9]+}}, 0x42c80000
define amdgpu_kernel void @tanh_f32_constant_100.0(ptr addrspace(1) %out) #1 {
  %tanh = call float @llvm.amdgcn.tanh.f32(float 100.0) #0
  store float %tanh, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}tanh_undef_f32:
; SDAG-NOT: v_tanh_f32
define amdgpu_kernel void @tanh_undef_f32(ptr addrspace(1) %out) #1 {
  %tanh = call float @llvm.amdgcn.tanh.f32(float undef)
  store float %tanh, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}tanh_f16:
; GCN: v_tanh_f16_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @tanh_f16(ptr addrspace(1) %out, half %src) #1 {
  %tanh = call half @llvm.amdgcn.tanh.f16(half %src) #0
  store half %tanh, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}tanh_f16_constant_4.0
; GCN: v_tanh_f16_e32 {{v[0-9]+}}, 4.0
define amdgpu_kernel void @tanh_f16_constant_4.0(ptr addrspace(1) %out) #1 {
  %tanh = call half @llvm.amdgcn.tanh.f16(half 4.0) #0
  store half %tanh, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}tanh_f16_constant_100.0
; GCN: v_tanh_f16_e32 {{v[0-9]+}}, 0x5640
define amdgpu_kernel void @tanh_f16_constant_100.0(ptr addrspace(1) %out) #1 {
  %tanh = call half @llvm.amdgcn.tanh.f16(half 100.0) #0
  store half %tanh, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}tanh_undef_f16:
; SDAG-NOT: v_tanh_f16
define amdgpu_kernel void @tanh_undef_f16(ptr addrspace(1) %out) #1 {
  %tanh = call half @llvm.amdgcn.tanh.f16(half undef)
  store half %tanh, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}tanh_bf16:
; GCN: v_tanh_bf16_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @tanh_bf16(ptr addrspace(1) %out, bfloat %src) #1 {
  %tanh = call bfloat @llvm.amdgcn.tanh.bf16(bfloat %src) #0
  store bfloat %tanh, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}tanh_bf16_constant_4
; GCN: v_tanh_bf16_e32 v0, 4.0
define amdgpu_kernel void @tanh_bf16_constant_4(ptr addrspace(1) %out) #1 {
  %tanh = call bfloat @llvm.amdgcn.tanh.bf16(bfloat 4.0) #0
  store bfloat %tanh, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}tanh_bf16_constant_100
; GCN: v_tanh_bf16_e32 {{v[0-9]+}}, 0x42c8
define amdgpu_kernel void @tanh_bf16_constant_100(ptr addrspace(1) %out) #1 {
  %tanh = call bfloat @llvm.amdgcn.tanh.bf16(bfloat 100.0) #0
  store bfloat %tanh, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}tanh_undef_bf16:
; SDAG-NOT: v_tanh_bf16
define amdgpu_kernel void @tanh_undef_bf16(ptr addrspace(1) %out) #1 {
  %tanh = call bfloat @llvm.amdgcn.tanh.bf16(bfloat undef)
  store bfloat %tanh, ptr addrspace(1) %out, align 2
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
