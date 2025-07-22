;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx90a  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx90a -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half>, <4 x half>, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x4f16(<4 x half>, <4 x half>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half>, <4 x half>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)
declare <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32, i32, <32 x i32>, i32, i32, i32)
declare <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32, i32, <16 x i32>, i32, i32, i32)
declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32, i32, i32)

define amdgpu_kernel void @test_mfma_f32_32x32x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <32 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 1.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_16x16x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 1.0, <16 x float> %in.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_4x4x1f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 1.0, <4 x float> %in.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_32x32x2f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float 1.0, float 1.0, <16 x float> %in.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_16x16x4f32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float 1.0, float 1.0, <4 x float> %in.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_32x32x4f16(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <32 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> poison, <4 x half> poison, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_16x16x4f16(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x4f16(<4 x half> poison, <4 x half> poison, <16 x float> %in.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_4x4x4f16(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half> poison, <4 x half> poison, <4 x float> %in.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_32x32x8f16(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> poison, <4 x half> poison, <16 x float> %in.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_f32_16x16x16f16(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> poison, <4 x half> poison, <4 x float> %in.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_i32_32x32x4i8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <32 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 1, i32 1, <32 x i32> %in.1, i32 0, i32 0, i32 0)
  store <32 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_i32_16x16x4i8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32 1, i32 1, <16 x i32> %in.1, i32 0, i32 0, i32 0)
  store <16 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}

define amdgpu_kernel void @test_mfma_i32_4x4x4i8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 1, <4 x i32> %in.1, i32 0, i32 0, i32 0)
  store <4 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}

attributes #0 = { "amdgpu-agpr-alloc"="0" }
