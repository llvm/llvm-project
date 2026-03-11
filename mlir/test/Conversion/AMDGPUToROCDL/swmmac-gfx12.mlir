// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1200 --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @rocdl.swmmac
func.func @rocdl.swmmac(
  %v32i4 : vector<32xi4>,
  %v16f16 : vector<16xf16>, %v16bf16 : vector<16xbf16>, %v16i8 : vector<16xi8>,
  %v16i4 : vector<16xi4>, %v16f8 : vector<16xf8E4M3FN>, %v16bf8 : vector<16xf8E5M2>,
  %v8f32 : vector<8xf32>, %v8i32 : vector<8xi32>, %v8f16 : vector<8xf16>, %v8bf16 : vector<8xbf16>, %v8i8 : vector<8xi8>,
  %v8i4 : vector<8xi4>, %v8f8 : vector<8xf8E4M3FN>, %v8bf8 : vector<8xf8E5M2>,
  %v4f32 : vector<4xf32>, %v4f16 : vector<4xf16>, %v4bf16 : vector<4xbf16>, %v4i8 : vector<4xi8>, %v4i32 : vector<4xi32>,
  %v4f8 : vector<4xf8E4M3FN>, %v4bf8 : vector<4xf8E5M2>,
  %idx : vector<4xi8>) {

  // Wave32

  // CHECK: rocdl.swmmac.f32.16x16x32.f16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<8xf16>, vector<16xf16>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_0 = amdgpu.sparse_wmma 16x16x32 %v8f16 * %v16f16 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf16>, vector<16xf16>, vector<8xf32>
  
  // CHECK: rocdl.swmmac.f32.16x16x32.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<8xi16>, vector<16xi16>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_1 = amdgpu.sparse_wmma 16x16x32 %v8bf16 * %v16bf16 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xbf16>, vector<16xbf16>, vector<8xf32>

  // CHECK: rocdl.swmmac.f16.16x16x32.f16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<8xf16>, vector<16xf16>, vector<8xf16>, i32) -> vector<8xf16>
  %w32_2 = amdgpu.sparse_wmma 16x16x32 %v8f16 * %v16f16 + %v8f16 sparse(%idx : vector<4xi8>) : vector<8xf16>, vector<16xf16>, vector<8xf16>

  // CHECK: rocdl.swmmac.bf16.16x16x32.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<8xi16>, vector<16xi16>, vector<8xi16>, i32) -> vector<8xi16>
  %w32_3 = amdgpu.sparse_wmma 16x16x32 %v8bf16 * %v16bf16 + %v8bf16 sparse(%idx : vector<4xi8>) : vector<8xbf16>, vector<16xbf16>, vector<8xbf16>

  // CHECK: rocdl.swmmac.i32.16x16x32.iu8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %w32_4 = amdgpu.sparse_wmma 16x16x32 %v8i8 * %v16i8 + %v8i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<8xi8>, vector<16xi8>, vector<8xi32>

  // CHECK: rocdl.swmmac.i32.16x16x32.iu4 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, vector<2xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %w32_5 = amdgpu.sparse_wmma 16x16x32 %v8i4 * %v16i4 + %v8i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<8xi4>, vector<16xi4>, vector<8xi32>

  // CHECK: rocdl.swmmac.i32.16x16x64.iu4 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %w32_6 = amdgpu.sparse_wmma 16x16x64 %v16i4 * %v32i4 + %v8i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<16xi4>, vector<32xi4>, vector<8xi32>

  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_7 = amdgpu.sparse_wmma 16x16x32 %v8f8 * %v16f8 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf8E4M3FN>, vector<16xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_8 = amdgpu.sparse_wmma 16x16x32 %v8f8 * %v16bf8 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf8E4M3FN>, vector<16xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_9 = amdgpu.sparse_wmma 16x16x32 %v8bf8 * %v16f8 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf8E5M2>, vector<16xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<2xi32>, vector<4xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_10 = amdgpu.sparse_wmma 16x16x32 %v8bf8 * %v16bf8 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf8E5M2>, vector<16xf8E5M2>, vector<8xf32>

  // Wave64

  // CHECK: rocdl.swmmac.f32.16x16x32.f16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<4xf16>, vector<8xf16>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_0 = amdgpu.sparse_wmma 16x16x32 %v4f16 * %v8f16 + %v4f32 sparse(%idx : vector<4xi8>) {wave64} : vector<4xf16>, vector<8xf16>, vector<4xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<4xi16>, vector<8xi16>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_1 = amdgpu.sparse_wmma 16x16x32 %v4bf16 * %v8bf16 + %v4f32 sparse(%idx : vector<4xi8>) {wave64} : vector<4xbf16>, vector<8xbf16>, vector<4xf32>

  // CHECK: rocdl.swmmac.f16.16x16x32.f16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<4xf16>, vector<8xf16>, vector<4xf16>, i32) -> vector<4xf16>
  %w64_2 = amdgpu.sparse_wmma 16x16x32 %v4f16 * %v8f16 + %v4f16 sparse(%idx : vector<4xi8>) {wave64} : vector<4xf16>, vector<8xf16>, vector<4xf16>

  // CHECK: rocdl.swmmac.bf16.16x16x32.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (vector<4xi16>, vector<8xi16>, vector<4xi16>, i32) -> vector<4xi16>
  %w64_3 = amdgpu.sparse_wmma 16x16x32 %v4bf16 * %v8bf16 + %v4bf16 sparse(%idx : vector<4xi8>) {wave64} : vector<4xbf16>, vector<8xbf16>, vector<4xbf16>

  // CHECK: rocdl.swmmac.i32.16x16x32.iu8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, vector<2xi32>, vector<4xi32>, i32) -> vector<4xi32>
  %w64_4 = amdgpu.sparse_wmma 16x16x32 %v4i8 * %v8i8 + %v4i32 sparse(%idx : vector<4xi8>) {wave64} : vector<4xi8>, vector<8xi8>, vector<4xi32>

  // CHECK: rocdl.swmmac.i32.16x16x32.iu4 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, i32, vector<4xi32>, i32) -> vector<4xi32>
  %w64_5 = amdgpu.sparse_wmma 16x16x32 %v8i4 * %v8i4 + %v4i32 sparse(%idx : vector<4xi8>) {wave64} : vector<8xi4>, vector<8xi4>, vector<4xi32>

  // CHECK: rocdl.swmmac.i32.16x16x64.iu4 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, vector<2xi32>, vector<4xi32>, i32) -> vector<4xi32>
  %w64_6 = amdgpu.sparse_wmma 16x16x64 %v8i4 * %v16i4 + %v4i32 sparse(%idx : vector<4xi8>) {wave64} : vector<8xi4>, vector<16xi4>, vector<4xi32>

  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, vector<2xi32>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_7 = amdgpu.sparse_wmma 16x16x32 %v4f8 * %v8f8 + %v4f32 sparse(%idx : vector<4xi8>) {wave64} : vector<4xf8E4M3FN>, vector<8xf8E4M3FN>, vector<4xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, vector<2xi32>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_8 = amdgpu.sparse_wmma 16x16x32 %v4f8 * %v8bf8 + %v4f32 sparse(%idx : vector<4xi8>) {wave64} : vector<4xf8E4M3FN>, vector<8xf8E5M2>, vector<4xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, vector<2xi32>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_9 = amdgpu.sparse_wmma 16x16x32 %v4bf8 * %v8f8 + %v4f32 sparse(%idx : vector<4xi8>) {wave64} : vector<4xf8E5M2>, vector<8xf8E4M3FN>, vector<4xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, vector<2xi32>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_10 = amdgpu.sparse_wmma 16x16x32 %v4bf8 * %v8bf8 + %v4f32 sparse(%idx : vector<4xi8>) {wave64} : vector<4xf8E5M2>, vector<8xf8E5M2>, vector<4xf32>

  func.return
}
