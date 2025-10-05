// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1200 --allow-unregistered-dialect | FileCheck %s
// CHECK-LABEL: @wmma_to_rocdl
func.func @wmma_to_rocdl(%arg0 : vector<8xf16>, %arg1 : vector<4xf16>,
                         %arg2 : vector<8xf32>, %arg3 : vector<4xf32>,
                         %arg4 : vector<8xbf16>, %arg5 : vector<4xbf16>,
                         %arg6 : vector<8xf8E4M3FN>, %arg7 : vector<4xf8E4M3FN>,
                         %arg8 : vector<8xf8E5M2>, %arg9 : vector<4xf8E5M2>,
                         %arg10 : vector<8xi8>, %arg11 : vector<4xi8>,
                         %arg12 : vector<8xi32>, %arg13 : vector<4xi32>,
                         %arg14 : vector<16xi4>, %arg15 : vector<8xi4>, %arg16 : vector<4xi4>) {
  // CHECK: rocdl.wmma.f32.16x16x16.f16{{.*}}: (vector<8xf16>, vector<8xf16>, vector<8xf32>) -> vector<8xf32>
  amdgpu.wmma %arg0 * %arg0 + %arg2 : vector<8xf16>, vector<8xf16>, vector<8xf32>
  // CHECK: rocdl.wmma.f32.16x16x16.f16{{.*}}: (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  amdgpu.wmma %arg1 * %arg1 + %arg3 : vector<4xf16>, vector<4xf16>, vector<4xf32>

  // CHECK: rocdl.wmma.f32.16x16x16.bf16{{.*}}: (vector<8xi16>, vector<8xi16>, vector<8xf32>) -> vector<8xf32>
  amdgpu.wmma %arg4 * %arg4 + %arg2 : vector<8xbf16>, vector<8xbf16>, vector<8xf32>
  // CHECK: rocdl.wmma.f32.16x16x16.bf16{{.*}}: (vector<4xi16>, vector<4xi16>, vector<4xf32>) -> vector<4xf32>
  amdgpu.wmma %arg5 * %arg5 + %arg3 : vector<4xbf16>, vector<4xbf16>, vector<4xf32>

  // CHECK: rocdl.wmma.f16.16x16x16.f16{{.*}}: (vector<8xf16>, vector<8xf16>, vector<8xf16>, i1) -> vector<8xf16>
  amdgpu.wmma %arg0 * %arg0 + %arg0 : vector<8xf16>, vector<8xf16>, vector<8xf16>
  // CHECK: rocdl.wmma.f16.16x16x16.f16{{.*}}: (vector<4xf16>, vector<4xf16>, vector<4xf16>, i1) -> vector<4xf16>
  amdgpu.wmma %arg1 * %arg1 + %arg1 : vector<4xf16>, vector<4xf16>, vector<4xf16>

  // CHECK: %[[raw_bf16x8:.+]] = rocdl.wmma.bf16.16x16x16.bf16{{.*}}: (vector<8xi16>, vector<8xi16>, vector<8xi16>, i1) -> vector<8xi16>
  // CHECK-NEXT: llvm.bitcast %[[raw_bf16x8]] : vector<8xi16> to vector<8xbf16>
  amdgpu.wmma %arg4 * %arg4 + %arg4 : vector<8xbf16>, vector<8xbf16>, vector<8xbf16>
  // CHECK: rocdl.wmma.bf16.16x16x16.bf16{{.*}}: (vector<4xi16>, vector<4xi16>, vector<4xi16>, i1) -> vector<4xi16>
  amdgpu.wmma %arg5 * %arg5 + %arg5 : vector<4xbf16>, vector<4xbf16>, vector<4xbf16>

  // CHECK: rocdl.wmma.f32.16x16x16.fp8_fp8{{.*}}: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>
  amdgpu.wmma %arg6 * %arg6 + %arg2 : vector<8xf8E4M3FN>, vector<8xf8E4M3FN>, vector<8xf32>
  // CHECK: rocdl.wmma.f32.16x16x16.fp8_fp8{{.*}}: (i32, i32, vector<4xf32>) -> vector<4xf32>
  amdgpu.wmma %arg7 * %arg7 + %arg3 : vector<4xf8E4M3FN>, vector<4xf8E4M3FN>, vector<4xf32>

  // CHECK: rocdl.wmma.f32.16x16x16.fp8_bf8{{.*}}: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>
  amdgpu.wmma %arg6 * %arg8 + %arg2 : vector<8xf8E4M3FN>, vector<8xf8E5M2>, vector<8xf32>
  // CHECK: rocdl.wmma.f32.16x16x16.fp8_bf8{{.*}}: (i32, i32, vector<4xf32>) -> vector<4xf32>
  amdgpu.wmma %arg7 * %arg9 + %arg3 : vector<4xf8E4M3FN>, vector<4xf8E5M2>, vector<4xf32>

  // CHECK: rocdl.wmma.f32.16x16x16.bf8_bf8{{.*}}: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>
  amdgpu.wmma %arg8 * %arg8 + %arg2 : vector<8xf8E5M2>, vector<8xf8E5M2>, vector<8xf32>
  // CHECK: rocdl.wmma.f32.16x16x16.bf8_bf8{{.*}}: (i32, i32, vector<4xf32>) -> vector<4xf32>
  amdgpu.wmma %arg9 * %arg9 + %arg3 : vector<4xf8E5M2>, vector<4xf8E5M2>, vector<4xf32>

  // CHECK: rocdl.wmma.f32.16x16x16.bf8_fp8{{.*}}: (vector<2xi32>, vector<2xi32>, vector<8xf32>) -> vector<8xf32>
  amdgpu.wmma %arg8 * %arg6 + %arg2 : vector<8xf8E5M2>, vector<8xf8E4M3FN>, vector<8xf32>
  // CHECK: rocdl.wmma.f32.16x16x16.bf8_fp8{{.*}}: (i32, i32, vector<4xf32>) -> vector<4xf32>
  amdgpu.wmma %arg9 * %arg7 + %arg3 : vector<4xf8E5M2>, vector<4xf8E4M3FN>, vector<4xf32>

  // CHECK: rocdl.wmma.i32.16x16x16.iu8{{.*}}: (i1, vector<2xi32>, i1, vector<2xi32>, vector<8xi32>, i1) -> vector<8xi32>
  amdgpu.wmma %arg10 * %arg10 + %arg12 {clamp} : vector<8xi8>, vector<8xi8>, vector<8xi32>
  // CHECK: rocdl.wmma.i32.16x16x16.iu8{{.*}}: (i1, i32, i1, i32, vector<4xi32>, i1) -> vector<4xi32>
  amdgpu.wmma %arg11 * %arg11 + %arg13 {unsignedA, unsignedB, clamp}: vector<4xi8>, vector<4xi8>, vector<4xi32>

  // CHECK: rocdl.wmma.i32.16x16x32.iu4{{.*}}: (i1, vector<2xi32>, i1, vector<2xi32>, vector<8xi32>, i1) -> vector<8xi32>
  amdgpu.wmma %arg14 * %arg14 + %arg12 {clamp} : vector<16xi4>, vector<16xi4>, vector<8xi32>
  // CHECK: rocdl.wmma.i32.16x16x32.iu4{{.*}}: (i1, i32, i1, i32, vector<4xi32>, i1) -> vector<4xi32>
  amdgpu.wmma %arg15 * %arg15 + %arg13 {clamp} : vector<8xi4>, vector<8xi4>, vector<4xi32>

  // CHECK: rocdl.wmma.i32.16x16x16.iu4{{.*}}: (i1, i32, i1, i32, vector<8xi32>, i1) -> vector<8xi32>
  amdgpu.wmma %arg15 * %arg15 + %arg12 {clamp} : vector<8xi4>, vector<8xi4>, vector<8xi32>
  // CHECK: rocdl.wmma.i32.16x16x16.iu4{{.*}}: (i1, i32, i1, i32, vector<4xi32>, i1) -> vector<4xi32>
  amdgpu.wmma %arg16 * %arg16 + %arg13 {clamp} : vector<4xi4>, vector<4xi4>, vector<4xi32>

  func.return
}
