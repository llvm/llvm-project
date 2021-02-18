; RUN: llc < %s -march=amdgcn -mcpu=gfx908 | FileCheck %s --check-prefixes=GCN,GFX908
; RUN: llc < %s -march=amdgcn -mcpu=gfx90a | FileCheck %s --check-prefixes=GCN,GFX90A

; GCN-LABEL: {{^}}mul_legacy
; GFX908: v_mul_legacy_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX90A: v_mul_legacy_f32_e64 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @mul_legacy(
    float addrspace(1)* %r,
    float addrspace(1)* %a,
    float addrspace(1)* %b) {
entry:
  %a.val = load volatile float, float addrspace(1)* %a
  %b.val = load volatile float, float addrspace(1)* %b
  %r.val = call float @llvm.pow.f32(float %a.val, float %b.val)
  store float %r.val, float addrspace(1)* %r
  ret void
}

declare float @llvm.pow.f32(float ,float ) readonly
