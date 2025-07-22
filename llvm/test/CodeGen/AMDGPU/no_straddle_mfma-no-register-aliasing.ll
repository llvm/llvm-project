;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx908  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx908 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)

define amdgpu_kernel void @test_mfma_f32_32x32x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <32 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %mai.1, i32 0, i32 0, i32 0)
  %tmp.1 = shufflevector <32 x float> %mai.2, <32 x float> %mai.1, <32 x i32> <i32 32, i32 33, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29>
  %mai.3 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %tmp.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.3, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_16x16x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %mai.1, i32 0, i32 0, i32 0)
  %tmp.1 = shufflevector <16 x float> %mai.2, <16 x float> %mai.1, <16 x i32> <i32 16, i32 17, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13>
  %mai.3 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %tmp.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.3, ptr addrspace(1) %arg
  ret void
}


define amdgpu_kernel void @test_mfma_f32_4x4x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %mai.1, i32 0, i32 0, i32 0)
  %tmp.1 = shufflevector <4 x float> %mai.1, <4 x float> %mai.2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %mai.3 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %tmp.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.3, ptr addrspace(1) %arg
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
