; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck -check-prefix=GFX942 %s

; Test the llvm.amdgcn.rcp.normal.f32 intrinsic which computes 1.0f / x
; using Newton-Raphson refinement. It assumes the input is a normal float.
; The expected lowering is:
;   y0 = v_rcp_f32(x)           ; initial approximation
;   err = v_fma_f32(x, y0, -1)  ; error = x * y0 - 1
;   y1 = v_fma_f32(y0, -err, y0) ; y1 = y0 + y0 * (1 - x * y0)

declare float @llvm.amdgcn.rcp.normal.f32(float) #0

; GFX9-LABEL: {{^}}test_rcp_f32_normal:
; GFX9: v_rcp_f32_e32 [[RCP:v[0-9]+]], s{{[0-9]+}}
; GFX9: v_fma_f32 [[ERR:v[0-9]+]], s{{[0-9]+}}, [[RCP]], -1.0
; GFX9: v_fma_f32 {{v[0-9]+}}, [[RCP]], -[[ERR]], [[RCP]]
;
; GFX10-LABEL: {{^}}test_rcp_f32_normal:
; GFX10: v_rcp_f32_e32 [[RCP:v[0-9]+]], s{{[0-9]+}}
; GFX10: v_fma_f32 [[ERR:v[0-9]+]], s{{[0-9]+}}, [[RCP]], -1.0
; GFX10: v_fma_f32 {{v[0-9]+}}, [[RCP]], -[[ERR]], [[RCP]]
;
; GFX942-LABEL: {{^}}test_rcp_f32_normal:
; GFX942: v_rcp_f32_e32 [[RCP:v[0-9]+]], s{{[0-9]+}}
; GFX942: v_fma_f32 [[ERR:v[0-9]+]], s{{[0-9]+}}, [[RCP]], -1.0
; GFX942: v_fma_f32 {{v[0-9]+}}, [[RCP]], -[[ERR]], [[RCP]]
define amdgpu_kernel void @test_rcp_f32_normal(ptr addrspace(1) %out, float %x) #1 {
  %rcp = call float @llvm.amdgcn.rcp.normal.f32(float %x)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; Test with a constant input - should fold to a constant
; GFX9-LABEL: {{^}}test_rcp_f32_normal_const:
; GFX9: v_mov_b32_e32 v{{[0-9]+}}, 0.5
;
; GFX10-LABEL: {{^}}test_rcp_f32_normal_const:
; GFX10: v_mov_b32_e32 v{{[0-9]+}}, 0.5
;
; GFX942-LABEL: {{^}}test_rcp_f32_normal_const:
; GFX942: v_mov_b32_e32 v{{[0-9]+}}, 0.5
define amdgpu_kernel void @test_rcp_f32_normal_const(ptr addrspace(1) %out) #1 {
  %rcp = call float @llvm.amdgcn.rcp.normal.f32(float 2.0)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; Test with another constant
; GFX9-LABEL: {{^}}test_rcp_f32_normal_const_10:
; GFX9: v_mov_b32_e32 v{{[0-9]+}}, 0x3dcccccd
;
; GFX10-LABEL: {{^}}test_rcp_f32_normal_const_10:
; GFX10: v_mov_b32_e32 v{{[0-9]+}}, 0x3dcccccd
;
; GFX942-LABEL: {{^}}test_rcp_f32_normal_const_10:
; GFX942: v_mov_b32_e32 v{{[0-9]+}}, 0x3dcccccd
define amdgpu_kernel void @test_rcp_f32_normal_const_10(ptr addrspace(1) %out) #1 {
  %rcp = call float @llvm.amdgcn.rcp.normal.f32(float 10.0)
  store float %rcp, ptr addrspace(1) %out, align 4
  ret void
}

; Test chained reciprocals (common pattern)
; GFX9-LABEL: {{^}}test_rcp_f32_normal_chain:
; GFX9: v_rcp_f32_e32
; GFX9: v_fma_f32
; GFX9: v_fma_f32
; GFX9: v_add_f32
; GFX9: v_rcp_f32_e32
; GFX9: v_fma_f32
; GFX9: v_fma_f32
;
; GFX10-LABEL: {{^}}test_rcp_f32_normal_chain:
; GFX10: v_rcp_f32_e32
; GFX10: v_fma_f32
; GFX10: v_fma_f32
; GFX10: v_add_f32
; GFX10: v_rcp_f32_e32
; GFX10: v_fma_f32
; GFX10: v_fma_f32
;
; GFX942-LABEL: {{^}}test_rcp_f32_normal_chain:
; GFX942: v_rcp_f32_e32
; GFX942: v_fma_f32
; GFX942: v_fma_f32
; GFX942: v_add_f32
; GFX942: v_rcp_f32_e32
; GFX942: v_fma_f32
; GFX942: v_fma_f32
define amdgpu_kernel void @test_rcp_f32_normal_chain(ptr addrspace(1) %out, float %x) #1 {
  %r1 = call float @llvm.amdgcn.rcp.normal.f32(float %x)
  %x2 = fadd float %x, %r1
  %r2 = call float @llvm.amdgcn.rcp.normal.f32(float %x2)
  store float %r2, ptr addrspace(1) %out, align 4
  ret void
}

; Test that the intrinsic produces a canonical result
; (important for correctly handling denormal outputs)
; GFX9-LABEL: {{^}}test_rcp_f32_normal_canonicalized:
; GFX9: v_rcp_f32_e32
; GFX9: v_fma_f32
; GFX9: v_fma_f32
; GFX9-NOT: v_max_f32
; GFX9-NOT: v_mul_f32{{.*}}1.0
;
; GFX10-LABEL: {{^}}test_rcp_f32_normal_canonicalized:
; GFX10: v_rcp_f32_e32
; GFX10: v_fma_f32
; GFX10: v_fma_f32
; GFX10-NOT: v_max_f32
; GFX10-NOT: v_mul_f32{{.*}}1.0
;
; GFX942-LABEL: {{^}}test_rcp_f32_normal_canonicalized:
; GFX942: v_rcp_f32_e32
; GFX942: v_fma_f32
; GFX942: v_fma_f32
; GFX942-NOT: v_max_f32
; GFX942-NOT: v_mul_f32{{.*}}1.0
define amdgpu_kernel void @test_rcp_f32_normal_canonicalized(ptr addrspace(1) %out, float %x) #1 {
  %rcp = call float @llvm.amdgcn.rcp.normal.f32(float %x)
  %canon = call float @llvm.canonicalize.f32(float %rcp)
  store float %canon, ptr addrspace(1) %out, align 4
  ret void
}

declare float @llvm.canonicalize.f32(float) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }
