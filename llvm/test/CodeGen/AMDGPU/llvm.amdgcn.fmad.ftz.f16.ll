; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8 %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

declare half @llvm.amdgcn.fmad.ftz.f16(half %a, half %b, half %c)

; GCN-LABEL: {{^}}mad_f16:
; GFX8: v_ma{{[dc]}}_f16
; GFX9: v_mad_legacy_f16
define amdgpu_kernel void @mad_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %r.val = call half @llvm.amdgcn.fmad.ftz.f16(half %a.val, half %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f16_imm_a:
; GCN: v_madmk_f16 {{v[0-9]+}}, {{v[0-9]+}}, 0x4800, {{v[0-9]+}}
define amdgpu_kernel void @mad_f16_imm_a(
    half addrspace(1)* %r,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %r.val = call half @llvm.amdgcn.fmad.ftz.f16(half 8.0, half %b.val, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f16_imm_b:
; GCN:  s_movk_i32 [[KB:s[0-9]+]], 0x4800
; GFX8: v_mad_f16 {{v[0-9]+}}, {{v[0-9]+}}, [[KB]],
; GFX9: v_mad_legacy_f16 {{v[0-9]+}}, {{v[0-9]+}}, [[KB]],
define amdgpu_kernel void @mad_f16_imm_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %c) {
  %a.val = load half, half addrspace(1)* %a
  %c.val = load half, half addrspace(1)* %c
  %r.val = call half @llvm.amdgcn.fmad.ftz.f16(half %a.val, half 8.0, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f16_imm_c:
; GCN: v_madak_f16 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, 0x4800{{$}}
define amdgpu_kernel void @mad_f16_imm_c(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = call half @llvm.amdgcn.fmad.ftz.f16(half %a.val, half %b.val, half 8.0)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f16_neg_b:
; GFX8: v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_mad_legacy_f16 v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @mad_f16_neg_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %neg.b = fsub half -0.0, %b.val
  %r.val = call half @llvm.amdgcn.fmad.ftz.f16(half %a.val, half %neg.b, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f16_abs_b:
; GFX8: v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}}
; GFX9: v_mad_legacy_f16 v{{[0-9]+}}, v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}}
define amdgpu_kernel void @mad_f16_abs_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %abs.b = call half @llvm.fabs.f16(half %b.val)
  %r.val = call half @llvm.amdgcn.fmad.ftz.f16(half %a.val, half %abs.b, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mad_f16_neg_abs_b:
; GFX8: v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, -|v{{[0-9]+}}|, v{{[0-9]+}}
; GFX9: v_mad_legacy_f16 v{{[0-9]+}}, v{{[0-9]+}}, -|v{{[0-9]+}}|, v{{[0-9]+}}
define amdgpu_kernel void @mad_f16_neg_abs_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %abs.b = call half @llvm.fabs.f16(half %b.val)
  %neg.abs.b = fsub half -0.0, %abs.b
  %r.val = call half @llvm.amdgcn.fmad.ftz.f16(half %a.val, half %neg.abs.b, half %c.val)
  store half %r.val, half addrspace(1)* %r
  ret void
}

declare half @llvm.fabs.f16(half)
