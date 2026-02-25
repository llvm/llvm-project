// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1200 --split-input-file --verify-diagnostics | FileCheck %s


func.func @rocdl.swmmac(
  %v32i4 : vector<32xi4>,
  %v16f16 : vector<16xf16>, %v16bf16 : vector<16xbf16>, %v16i8 : vector<16xi8>,
  %v16i4 : vector<16xi4>, %v16f8 : vector<16xf8E4M3FN>, %v16bf8 : vector<16xf8E5M2>,
  %v8f32 : vector<8xf32>, %v8i32 : vector<8xi32>, %v8f16 : vector<8xf16>, %v8bf16 : vector<8xbf16>, %v8i8 : vector<8xi8>,
  %v8i4 : vector<8xi4>, %v8f8 : vector<8xf8E4M3FN>, %v8bf8 : vector<8xf8E5M2>,
  %idx : vector<4xi8>) {

  // ---- Wave32 -----

  // CHECK: rocdl.swmmac.f32.16x16x32.f16 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<8xf16>, vector<16xf16>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_0 = amdgpu.sparse_wmma 16x16x32 %v8f16 * %v16f16 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf16>, vector<16xf16>, vector<8xf32>
  
  // CHECK: rocdl.swmmac.f32.16x16x32.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<8xi16>, vector<16xi16>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_1 = amdgpu.sparse_wmma 16x16x32 %v8bf16 * %v16bf16 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xbf16>, vector<16xbf16>, vector<8xf32>

  // CHECK: rocdl.swmmac.f16.16x16x32.f16 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<8xf16>, vector<16xf16>, vector<8xf16>, i32) -> vector<8xf16>
  %w32_2 = amdgpu.sparse_wmma 16x16x32 %v8bf16 * %v16bf16 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xbf16>, vector<16xbf16>, vector<8xf32>

  // CHECK: rocdl.swmmac.bf16.16x16x32.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<8xi16>, vector<16xi16>, vector<8xi16>, i32) -> vector<8xi16>
  %w32_3 = amdgpu.sparse_wmma 16x16x32 %v8bf16 * %v16bf16 + %v8bf16 sparse(%idx : vector<4xi8>) : vector<8xbf16>, vector<16xbf16>, vector<8xbf16>

  // CHECK: rocdl.swmmac.i32.16x16x32.iu8 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, clamp = true} : (vector<2xi32>, vector<4xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %w32_4 = amdgpu.sparse_wmma 16x16x32 %v8i8 * %v16i8 + %v8i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<8xi8>, vector<16xi8>, vector<8xi32>

  // CHECK: rocdl.swmmac.i32.16x16x32.iu4 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, clamp = true} : (i32, vector<2xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %w32_5 = amdgpu.sparse_wmma 16x16x32 %v8i4 * %v16i4 + %v8i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<8xi4>, vector<16xi4>, vector<8xi32>

  // CHECK: rocdl.swmmac.i32.16x16x64.iu4 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, clamp = true} : (vector<2xi32>, vector<4xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %w32_6 = amdgpu.sparse_wmma 16x16x64 %v16i4 * %v32i4 + %v8i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<16xi4>, vector<32xi4>, vector<8xi32>

  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<2xi32>, vector<4xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_7 = amdgpu.sparse_wmma 16x16x32 %v8f8 * %v16f8 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf8E4M3FN>, vector<16xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<2xi32>, vector<4xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_8 = amdgpu.sparse_wmma 16x16x32 %v8f8 * %v16bf8 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf8E4M3FN>, vector<16xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<2xi32>, vector<4xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_9 = amdgpu.sparse_wmma 16x16x32 %v8bf8 * %v16f8 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf8E5M2>, vector<16xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<2xi32>, vector<4xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_10 = amdgpu.sparse_wmma 16x16x32 %v8bf8 * %v16bf8 + %v8f32 sparse(%idx : vector<4xi8>) : vector<8xf8E5M2>, vector<16xf8E5M2>, vector<8xf32>

  func.return
}
