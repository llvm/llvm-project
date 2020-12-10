; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX908 %s

; GCN-LABEL: {{^}}accvgpr_write_read:
; GFX908: v_accvgpr_write [[AREG:a[0-9]+]], 1
; GFX908: v_accvgpr_read [[VREG:v[0-9]+]], [[AREG]]
; GFX908: global_store_dword v{{[0-9]+}}, [[VREG]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @accvgpr_write_read(float addrspace(1)* %arg) {
bb:
  %in.1 = load float, float addrspace(1)* %arg
  %init = tail call float asm "v_accvgpr_write $0, 1", "=a"()
  %read = tail call float asm "v_accvgpr_read $0, $1", "=v,a"(float %init)
  store float %read, float addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}v_mfma_f32_4x4x1f32_avva
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_mfma_f32_4x4x1f32 a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9:]+}}]
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
define amdgpu_kernel void @v_mfma_f32_4x4x1f32_avva(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> asm "v_mfma_f32_4x4x1f32 $0, $1, $2, $3", "=a,v,v,a"(float 1.0, float 2.0, <4 x float> %in.1)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}v_mfma_f32_4x4x1f32_aaaa
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_mfma_f32_4x4x1f32 a[{{[0-9:]+}}], a{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
define amdgpu_kernel void @v_mfma_f32_4x4x1f32_aaaa(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> asm "v_mfma_f32_4x4x1f32 $0, $1, $2, $3", "=a,a,a,a"(float 1.0, float 2.0, <4 x float> %in.1)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}v_mfma_f32_4x4x4f16_aaaa
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_mfma_f32_4x4x4f16 a[{{[0-9:]+}}], a[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}], a[{{[0-9:]+}}]
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
define amdgpu_kernel void @v_mfma_f32_4x4x4f16_aaaa(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> asm "v_mfma_f32_4x4x4f16 $0, $1, $2, $3", "=a,a,a,a"(<4 x half> <half 0xH3800, half 0xH3800, half 0xH3800, half 0xH3800>, <4 x half> <half 0xH03FF, half 0xH03FF, half 0xH03FF, half 0xH03FF>, <4 x float> %in.1)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}v_mfma_f32_16x16x1f32_avaa
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_mfma_f32_16x16x1f32 a[{{[0-9:]+}}], v{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
define amdgpu_kernel void @v_mfma_f32_16x16x1f32_avaa(<16 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> asm "v_mfma_f32_16x16x1f32 $0, $1, $2, $3", "=a,v,a,a"(float 1.0, float 2.0, <16 x float> %in.1)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}v_mfma_f32_32x32x1f32_avaa
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_accvgpr_write_b32
; GFX908: v_mfma_f32_32x32x1f32 a[{{[0-9:]+}}], v{{[0-9]+}}, a{{[0-9]+}}, a[{{[0-9:]+}}]
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
; GFX908: v_accvgpr_read_b32
define amdgpu_kernel void @v_mfma_f32_32x32x1f32_avaa(<32 x i32> addrspace(1)* %arg) {
bb:
  %in.1 = load <32 x i32>, <32 x i32> addrspace(1)* %arg
  %mai.1 = tail call <32 x i32> asm "v_mfma_f32_32x32x1f32 $0, $1, $2, $3", "=a,v,a,a"(float 1.0, float 2.0, <32 x i32> %in.1)
  store <32 x i32> %mai.1, <32 x i32> addrspace(1)* %arg
  ret void
}
