; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

declare float @llvm.amdgcn.fmad.ftz.f32(float %a, float %b, float %c)

; GCN-LABEL: {{^}}mad_f32:
; GCN:  v_ma{{[dc]}}_f32
define amdgpu_kernel void @mad_f32(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %b,
    float addrspace(1)* %c) {
  %a.val = load float, float addrspace(1)* %a
  %b.val = load float, float addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %b.val, float %c.val)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_imm_a:
; GCN: v_madmk_f32 {{v[0-9]+}}, {{v[0-9]+}}, 0x41000000,
define amdgpu_kernel void @mad_f32_imm_a(
    float addrspace(1)* %r,
    float addrspace(1)* %b,
    float addrspace(1)* %c) {
  %b.val = load float, float addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float 8.0, float %b.val, float %c.val)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_imm_b:
; GCN: v_mov_b32_e32 [[KB:v[0-9]+]], 0x41000000
; GCN:  v_ma{{[dc]}}_f32 {{v[0-9]+}}, {{[vs][0-9]+}}, [[KB]],
define amdgpu_kernel void @mad_f32_imm_b(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %c) {
  %a.val = load float, float addrspace(1)* %a
  %c.val = load float, float addrspace(1)* %c
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float 8.0, float %c.val)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_imm_c:
; GCN: v_mov_b32_e32 [[KC:v[0-9]+]], 0x41000000
; GCN:  v_ma{{[dc]}}_f32 {{v[0-9]+}}, {{[vs][0-9]+}}, {{v[0-9]+}}, [[KC]]{{$}}
define amdgpu_kernel void @mad_f32_imm_c(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %b) {
  %a.val = load float, float addrspace(1)* %a
  %b.val = load float, float addrspace(1)* %b
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %b.val, float 8.0)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_neg_b:
; GCN:  v_mad_f32 v{{[0-9]+}}, s{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @mad_f32_neg_b(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %b,
    float addrspace(1)* %c) {
  %a.val = load float, float addrspace(1)* %a
  %b.val = load float, float addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %neg.b = fsub float -0.0, %b.val
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %neg.b, float %c.val)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_abs_b:
; GCN:  v_mad_f32 v{{[0-9]+}}, s{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}}
define amdgpu_kernel void @mad_f32_abs_b(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %b,
    float addrspace(1)* %c) {
  %a.val = load float, float addrspace(1)* %a
  %b.val = load float, float addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %abs.b = call float @llvm.fabs.f32(float %b.val)
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %abs.b, float %c.val)
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_neg_abs_b:
; GCN:  v_mad_f32 v{{[0-9]+}}, s{{[0-9]+}}, -|v{{[0-9]+}}|, v{{[0-9]+}}
define amdgpu_kernel void @mad_f32_neg_abs_b(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %b,
    float addrspace(1)* %c) {
  %a.val = load float, float addrspace(1)* %a
  %b.val = load float, float addrspace(1)* %b
  %c.val = load float, float addrspace(1)* %c
  %abs.b = call float @llvm.fabs.f32(float %b.val)
  %neg.abs.b = fsub float -0.0, %abs.b
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %neg.abs.b, float %c.val)
  store float %r.val, float addrspace(1)* %r
  ret void
}

declare float @llvm.fabs.f32(float)
