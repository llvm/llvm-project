; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)

; GCN-LABEL: {{^}}test_32_agprs:
; GCN: v_mfma_f32_32x32x1f32 a[0:31], {{v[0-9]+}}, {{v[0-9]+}}, 0
; GCN-NOT: v28
; GCN: NumVgprs: 32
; GCN: VGPRBlocks: 7
define amdgpu_kernel void @test_32_agprs(<32 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> <float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}
