; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

declare void @llvm.amdgcn.s.setprio(i16)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32:
; GCN: s_setprio 1
; GCN: v_mfma
; GCN: v_mfma
; GCN: s_setprio 0
define amdgpu_kernel void @test_mfma_f32_4x4x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  call void @llvm.amdgcn.s.setprio(i16 1)
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 3.0, float 4.0, <4 x float> %mai.1, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.s.setprio(i16 0)
  store <4 x float> %mai.2, ptr addrspace(1) %arg
  ret void
}
