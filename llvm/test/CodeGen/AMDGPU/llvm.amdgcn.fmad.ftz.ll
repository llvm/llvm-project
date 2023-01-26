; RUN: llc -march=amdgcn -mcpu=tahiti -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -denormal-fp-math-f32=preserve-sign -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -denormal-fp-math-f32=ieee -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -denormal-fp-math-f32=ieee -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

declare float @llvm.amdgcn.fmad.ftz.f32(float %a, float %b, float %c)

; GCN-LABEL: {{^}}mad_f32:
; GCN:  v_ma{{[dc]}}_f32
define amdgpu_kernel void @mad_f32(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
  %a.val = load float, ptr addrspace(1) %a
  %b.val = load float, ptr addrspace(1) %b
  %c.val = load float, ptr addrspace(1) %c
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %b.val, float %c.val)
  store float %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_imm_a:
; GCN: v_madmk_f32 {{v[0-9]+}}, {{v[0-9]+}}, 0x41000000,
define amdgpu_kernel void @mad_f32_imm_a(
    ptr addrspace(1) %r,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
  %b.val = load float, ptr addrspace(1) %b
  %c.val = load float, ptr addrspace(1) %c
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float 8.0, float %b.val, float %c.val)
  store float %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_imm_b:
; GCN: v_mov_b32_e32 [[KB:v[0-9]+]], 0x41000000
; GCN: v_mac_f32_e32 {{v[0-9]+}}, {{[s][0-9]+}}, [[KB]]
define amdgpu_kernel void @mad_f32_imm_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %c) {
  %a.val = load float, ptr addrspace(1) %a
  %c.val = load float, ptr addrspace(1) %c
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float 8.0, float %c.val)
  store float %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_imm_c:
; GCN: v_mov_b32_e32 [[C:v[0-9]+]], 0x41000000
; GCN: s_load_dword [[B:s[0-9]+]]
; GCN: s_load_dword [[A:s[0-9]+]]
; GCN: v_mov_b32_e32 [[VB:v[0-9]+]], [[B]]
; GCN: v_mac_f32_e32 [[C]], {{s[0-9]+}}, [[VB]]{{$}}
define amdgpu_kernel void @mad_f32_imm_c(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) {
  %a.val = load float, ptr addrspace(1) %a
  %b.val = load float, ptr addrspace(1) %b
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %b.val, float 8.0)
  store float %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_neg_b:
; GCN:  v_mad_f32 v{{[0-9]+}}, s{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @mad_f32_neg_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
  %a.val = load float, ptr addrspace(1) %a
  %b.val = load float, ptr addrspace(1) %b
  %c.val = load float, ptr addrspace(1) %c
  %neg.b = fneg float %b.val
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %neg.b, float %c.val)
  store float %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_abs_b:
; GCN:  v_mad_f32 v{{[0-9]+}}, s{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}}
define amdgpu_kernel void @mad_f32_abs_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
  %a.val = load float, ptr addrspace(1) %a
  %b.val = load float, ptr addrspace(1) %b
  %c.val = load float, ptr addrspace(1) %c
  %abs.b = call float @llvm.fabs.f32(float %b.val)
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %abs.b, float %c.val)
  store float %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mad_f32_neg_abs_b:
; GCN:  v_mad_f32 v{{[0-9]+}}, s{{[0-9]+}}, -|v{{[0-9]+}}|, v{{[0-9]+}}
define amdgpu_kernel void @mad_f32_neg_abs_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
  %a.val = load float, ptr addrspace(1) %a
  %b.val = load float, ptr addrspace(1) %b
  %c.val = load float, ptr addrspace(1) %c
  %abs.b = call float @llvm.fabs.f32(float %b.val)
  %neg.abs.b = fneg float %abs.b
  %r.val = call float @llvm.amdgcn.fmad.ftz.f32(float %a.val, float %neg.abs.b, float %c.val)
  store float %r.val, ptr addrspace(1) %r
  ret void
}

declare float @llvm.fabs.f32(float)
