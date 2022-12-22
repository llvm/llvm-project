; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck --enable-var-scope --check-prefixes=GCN %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck --enable-var-scope --check-prefixes=GCN %s

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16>, <2 x i16>, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16>, <2 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16>, <2 x i16>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16>, <2 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16>, <2 x i16>, <4 x float>, i32, i32, i32)
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16.1k(<4 x i16>, <4 x i16>, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x4bf16.1k(<4 x i16>, <4 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8bf16.1k(<4 x i16>, <4 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32, i32, i32)
declare <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double, double, <4 x double>, i32, i32, i32)
declare double @llvm.amdgcn.mfma.f64.4x4x4f64(double, double, double, i32, i32, i32)
declare <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32, i32, <16 x i32>, i32, i32, i32)
declare <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32, i32, <4 x i32>, i32, i32, i32)

; GCN-LABEL: {{^}}test_mfma_f32_32x32x2bf16:
; GCN:  v_mfma_f32_32x32x2bf16 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x2bf16(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <32 x float>, ptr addrspace(1) %arg
  %a = bitcast i32 1 to <2 x i16>
  %b = bitcast i32 2 to <2 x i16>
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16> %a, <2 x i16> %b, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x2bf16:
; GCN: v_mfma_f32_16x16x2bf16 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x2bf16(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16> undef, <2 x i16> undef, <16 x float> %in.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x2bf16:
; GCN: v_mfma_f32_4x4x2bf16 v[{{[0-9:]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_4x4x2bf16(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> undef, <2 x i16> undef, <4 x float> %in.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x4bf16:
; GCN: v_mfma_f32_32x32x4bf16 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x4bf16(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16> undef, <2 x i16> undef, <16 x float> %in.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x8bf16:
; GCN: v_mfma_f32_16x16x8bf16 v[{{[0-9:]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x8bf16(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16> undef, <2 x i16> undef, <4 x float> %in.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x4bf16_1k:
; GCN:      v_mfma_f32_32x32x4bf16_1k v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x4bf16_1k(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <32 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16.1k(<4 x i16> undef, <4 x i16> undef, <32 x float> %in.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x4bf16_1k:
; GCN: v_mfma_f32_16x16x4bf16_1k v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x4bf16_1k(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x4bf16.1k(<4 x i16> undef, <4 x i16> undef, <16 x float> %in.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x4bf16_1k:
; GCN: v_mfma_f32_4x4x4bf16_1k v[{{[0-9:]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_4x4x4bf16_1k(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4bf16.1k(<4 x i16> undef, <4 x i16> undef, <4 x float> %in.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x8bf16_1k:
; GCN: v_mfma_f32_32x32x8bf16_1k v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x8bf16_1k(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8bf16.1k(<4 x i16> undef, <4 x i16> undef, <16 x float> %in.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x16bf16_1k:
; GCN: v_mfma_f32_16x16x16bf16_1k v[{{[0-9:]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x16bf16_1k(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> undef, <4 x i16> undef, <4 x float> %in.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f64_4x4x4f64:
; GCN: v_mfma_f64_4x4x4f64 v[{{[0-9:]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9:]+:[0-9]+}}
define amdgpu_kernel void @test_mfma_f64_4x4x4f64(ptr addrspace(1) %arg) {
bb:
  %mai.1 = tail call double @llvm.amdgcn.mfma.f64.4x4x4f64(double 1.0, double 1.0, double 128.0, i32 0, i32 0, i32 0)
  store double %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f64_16x16x4f64:
; GCN: v_mfma_f64_16x16x4f64 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_f64_16x16x4f64(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <4 x double>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double 1.0, double 1.0, <4 x double> %in.1, i32 0, i32 0, i32 0)
  store <4 x double> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_32x32x8i8:
; GCN: v_mfma_i32_32x32x8i8 v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_i32_32x32x8i8(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <16 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32 1, i32 1, <16 x i32> %in.1, i32 0, i32 0, i32 0)
  store <16 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_16x16x16i8:
; GCN: v_mfma_i32_16x16x16i8 v[{{[0-9:]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test_mfma_i32_16x16x16i8(ptr addrspace(1) %arg) {
bb:
  %in.1 = load <4 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32 1, i32 1, <4 x i32> %in.1, i32 0, i32 0, i32 0)
  store <4 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}
