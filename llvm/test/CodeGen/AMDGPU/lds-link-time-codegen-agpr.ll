; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -amdgpu-enable-object-linking < %s | FileCheck %s

; Verify that .amdgpu_num_agpr IS emitted when AGPRs are used on a target
; that supports them (gfx908 has a separate AGPR file).

declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)

define void @func_with_agpr(float %a, float %b, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float %a, float %b, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @kern(float %a, float %b, ptr addrspace(1) %out) {
  call void @func_with_agpr(float %a, float %b, ptr addrspace(1) %out)
  ret void
}

; CHECK:      .amdgpu_info func_with_agpr
; CHECK:        .amdgpu_num_agpr {{[1-9][0-9]*}}
; CHECK:      .end_amdgpu_info
; CHECK:      .amdgpu_info kern
; CHECK:      .end_amdgpu_info
