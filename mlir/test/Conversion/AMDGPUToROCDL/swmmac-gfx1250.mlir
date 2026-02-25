
// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1250 --split-input-file --verify-diagnostics | FileCheck %s

func.func @rocdl.swmmac(
  %v64i8 : vector<64xi8>, %v64f8 : vector<64xf8E4M3FN>, %v64bf8 : vector<64xf8E5M2>,
  %v32f16 : vector<32xf16>, %v32bf16 : vector<32xbf16>, %v32i8 : vector<32xi8>, %v32i4 : vector<32xi4>, %v32f8 : vector<32xf8E4M3FN>, %v32bf8 : vector<32xf8E5M2>,
  %v16f16 : vector<16xf16>, %v16bf16 : vector<16xbf16>, %v16f8 : vector<16xf8E4M3FN>, %v16bf8 : vector<16xf8E5M2>,
  %v8f32 : vector<8xf32>, %v8i32 : vector<8xi32>, %v8f16 : vector<8xf16>, %v8bf16 : vector<8xbf16>, %v8i8 : vector<8xi8>, %v8i4 : vector<8xi4>,
  %v4f32 : vector<4xf32>, %v4f16 : vector<4xf16>, %v4bf16 : vector<4xbf16>, %v4i32 : vector<4xi32>, %v4i8 : vector<4xi8>,
  %v4f8 : vector<4xf8E4M3FN>, %v4bf8 : vector<4xf8E5M2>,
  %idx : vector<4xi8>) {

  // ---- Wave32 -----

  // CHECK: rocdl.swmmac.f32.16x16x64.f16 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, reuseA = true, reuseB = true} : (vector<16xf16>, vector<32xf16>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_11 = amdgpu.sparse_wmma 16x16x64 %v16f16 * %v32f16 + %v8f32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, reuseA, reuseB} : vector<16xf16>, vector<32xf16>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x64.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, reuseA = true, reuseB = true} : (vector<16xbf16>, vector<32xbf16>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_12 = amdgpu.sparse_wmma 16x16x64 %v16bf16 * %v32bf16 + %v8f32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, reuseA, reuseB} : vector<16xbf16>, vector<32xbf16>, vector<8xf32>

  // CHECK: rocdl.swmmac.f16.16x16x64.f16 %v{{.*}}, %v{{.*}}, %v{{.*}}, %index {signA = true, signB = true, reuseA = true, reuseB = true} : (vector<16xf16>, vector<32xf16>, vector<8xf16>, i32) -> vector<8xf16>
  %w32_13 = amdgpu.sparse_wmma 16x16x64 %v16f16 * %v32f16 + %v8f16 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, reuseA, reuseB} : vector<16xf16>, vector<32xf16>, vector<8xf16>

  // CHECK: rocdl.swmmac.bf16.16x16x64.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, reuseA = true, reuseB = true} : (vector<16xbf16>, vector<32xbf16>, vector<8xbf16>, i32) -> vector<8xbf16>
  %w32_14 = amdgpu.sparse_wmma 16x16x64 %v16bf16 * %v32bf16 + %v8bf16 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, reuseA, reuseB} : vector<16xbf16>, vector<32xbf16>, vector<8xbf16>
 
  // CHECK:
   // f32 output ?? %15 = amdgpu.sparse_wmma 16x16x64 %v16bf16 * %v32bf16 + %v8bf16 sparse(%idx : vector<4xi8>) {signA = false, signB = false} : vector<16xbf16>, vector<32xbf16>, vector<8xbf16>
//  %w32_15 = rocdl.swmmac.bf16f32.16x16x64.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = false, signB = false, reuseA = false, reuseB = false} : (vector<16xbf16>, vector<32xbf16>, vector<8xbf16>, i32) -> vector<8xbf16>

  // CHECK: rocdl.swmmac.f32.16x16x128.fp8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %index {reuseA = true, reuseB = true} : (vector<8xi32>, vector<16xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_16 = amdgpu.sparse_wmma 16x16x128 %v32f8 * %v64f8 + %v8f32 sparse(%idx : vector<4xi8>) {reuseA, reuseB} : vector<32xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x128.fp8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %index {reuseA = true, reuseB = true} : (vector<8xi32>, vector<16xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_17 = amdgpu.sparse_wmma 16x16x128 %v32f8 * %v64bf8 + %v8f32 sparse(%idx : vector<4xi8>) {reuseA, reuseB} : vector<32xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x128.bf8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %index {reuseA = true, reuseB = true} : (vector<8xi32>, vector<16xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_18 = amdgpu.sparse_wmma 16x16x128 %v32bf8 * %v64f8 + %v8f32 sparse(%idx : vector<4xi8>) {reuseA, reuseB} : vector<32xf8E5M2>, vector<64xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.swmmac.f32.16x16x128.bf8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %index {reuseA = true, reuseB = true} : (vector<8xi32>, vector<16xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %w32_19 = amdgpu.sparse_wmma 16x16x128 %v32bf8 * %v64bf8 + %v8f32 sparse(%idx : vector<4xi8>) {reuseA, reuseB} : vector<32xf8E5M2>, vector<64xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.swmmac.f16.16x16x128.fp8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %index {reuseA = true, reuseB = true} : (vector<8xi32>, vector<16xi32>, vector<8xf16>, i32) -> vector<8xf16>
  %w32_20 = amdgpu.sparse_wmma 16x16x128 %v32f8 * %v64f8 + %v8f16 sparse(%idx : vector<4xi8>) {reuseA, reuseB} : vector<32xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf16>

  // CHECK: rocdl.swmmac.f16.16x16x128.fp8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %index {reuseA = true, reuseB = true} : (vector<8xi32>, vector<16xi32>, vector<8xf16>, i32) -> vector<8xf16>
  %w32_21 = amdgpu.sparse_wmma 16x16x128 %v32f8 * %v64bf8 + %v8f16 sparse(%idx : vector<4xi8>) {reuseA, reuseB} : vector<32xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.swmmac.f16.16x16x128.bf8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %index {reuseA = true, reuseB = true} : (vector<8xi32>, vector<16xi32>, vector<8xf16>, i32) -> vector<8xf16>
  %w32_22 = amdgpu.sparse_wmma 16x16x128 %v32bf8 * %v64f8 + %v8f16 sparse(%idx : vector<4xi8>) {reuseA, reuseB} : vector<32xf8E5M2>, vector<64xf8E4M3FN>, vector<8xf16>

  // CHECK: rocdl.swmmac.f16.16x16x128.bf8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %index {reuseA = true, reuseB = true} : (vector<8xi32>, vector<16xi32>, vector<8xf16>, i32) -> vector<8xf16>
  %w32_23 = amdgpu.sparse_wmma 16x16x128 %v32bf8 * %v64bf8 + %v8f16 sparse(%idx : vector<4xi8>) {reuseA, reuseB} : vector<32xf8E5M2>, vector<64xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.swmmac.i32.16x16x128.iu8 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, reuseA = true, reuseB = true, clamp = true} : (vector<8xi32>, vector<16xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %w32_24 = amdgpu.sparse_wmma 16x16x128 %v32i8 * %v64i8 + %v8i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, reuseA, reuseB, clamp} : vector<32xi8>, vector<64xi8>, vector<8xi32>

  // ---- Wave64 -----

  // CHECK: rocdl.swmmac.f32.16x16x32.f16 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<4xf16>, vector<8xf16>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_0 = amdgpu.sparse_wmma 16x16x32 %v4f16 * %v8f16 + %v4f32 sparse(%idx : vector<4xi8>) : vector<4xf16>, vector<8xf16>, vector<4xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<4xi16>, vector<8xi16>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_1 = amdgpu.sparse_wmma 16x16x32 %v4bf16 * %v8bf16 + %v4f32 sparse(%idx : vector<4xi8>) : vector<4xbf16>, vector<8xbf16>, vector<4xf32>

  // CHECK: rocdl.swmmac.f16.16x16x32.f16 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<4xf16>, vector<8xf16>, vector<4xf16>, i32) -> vector<4xf16>
  %w64_2 = amdgpu.sparse_wmma 16x16x32 %v4f16 * %v8f16 + %v4f16 sparse(%idx : vector<4xi8>) : vector<4xf16>, vector<8xf16>, vector<4xf16>

  // CHECK: rocdl.swmmac.bf16.16x16x32.bf16 %{{.*}}, %{{.*}}, %{{.*}}, %index : (vector<4xi16>, vector<8xi16>, vector<4xi16>, i32) -> vector<4xi16>
  %w64_3 = amdgpu.sparse_wmma 16x16x32 %v4bf16 * %v8bf16 + %v4bf16 sparse(%idx : vector<4xi8>) : vector<4xbf16>, vector<8xbf16>, vector<4xbf16>

  // CHECK: rocdl.swmmac.i32.16x16x32.iu8 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, clamp = true} : (i32, vector<2xi32>, vector<4xi32>, i32) -> vector<4xi32>
  %w64_4 = amdgpu.sparse_wmma 16x16x32 %v4i8 * %v8i8 + %v4i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<4xi8>, vector<8xi8>, vector<4xi32>

  // CHECK: rocdl.swmmac.i32.16x16x32.iu4 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, clamp = true} : (i32, i32, vector<4xi32>, i32) -> vector<4xi32>
  %w64_5 = amdgpu.sparse_wmma 16x16x32 %v8i4 * %v8i4 + %v4i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<8xi4>, vector<8xi4>, vector<4xi32>

  // CHECK: rocdl.swmmac.i32.16x16x64.iu4 %{{.*}}, %{{.*}}, %{{.*}}, %index {signA = true, signB = true, clamp = true} : (i32, vector<2xi32>, vector<4xi32>, i32) -> vector<4xi32>
  %w64_6 = amdgpu.sparse_wmma 16x16x64 %v8i4 * %v32i4 + %v4i32 sparse(%idx : vector<4xi8>) {unsignedA, unsignedB, clamp} : vector<8xi4>, vector<32xi4>, vector<4xi32>

  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %index : (i32, vector<2xi32>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_7 = amdgpu.sparse_wmma 16x16x32 %v4f8 * %v16f8 + %v4f32 sparse(%idx : vector<4xi8>) : vector<4xf8E4M3FN>, vector<16xf8E4M3FN>, vector<4xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.fp8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %index : (i32, vector<2xi32>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_8 = amdgpu.sparse_wmma 16x16x32 %v4f8 * %v16bf8 + %v4f32 sparse(%idx : vector<4xi8>) : vector<4xf8E4M3FN>, vector<16xf8E5M2>, vector<4xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.fp8 %{{.*}}, %{{.*}}, %{{.*}}, %index : (i32, vector<2xi32>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_9 = amdgpu.sparse_wmma 16x16x32 %v4bf8 * %v16f8 + %v4f32 sparse(%idx : vector<4xi8>) : vector<4xf8E5M2>, vector<16xf8E4M3FN>, vector<4xf32>

  // CHECK: rocdl.swmmac.f32.16x16x32.bf8.bf8 %{{.*}}, %{{.*}}, %{{.*}}, %index : (i32, vector<2xi32>, vector<4xf32>, i32) -> vector<4xf32>
  %w64_10 = amdgpu.sparse_wmma 16x16x32 %v4bf8 * %v16bf8 + %v4f32 sparse(%idx : vector<4xi8>) : vector<4xf8E5M2>, vector<16xf8E5M2>, vector<4xf32>

  func.return
}
