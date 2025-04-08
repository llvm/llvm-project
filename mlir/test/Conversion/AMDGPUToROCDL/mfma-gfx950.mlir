// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx950 -cse | FileCheck %s
func.func @mfma_to_rocdl(%arg0 : vector<8xf16>, %arg1 : vector<16xf32>,
                    %arg2 : vector<4xf32>, %arg3 : vector<8xbf16>,
                    %arg4 : vector<16xi8>, %arg5 : vector<16xi32>,
                    %arg6 : vector<4xi32>, %arg7 : vector<32xf8E4M3FN>,
                    %arg8 : vector<32xf8E5M2>, %arg9 : vector<32xf6E2M3FN>,
                    %arg10 : vector<32xf6E3M2FN>, %arg11 : vector<32xf4E2M1FN>) {
  // CHECK: %[[c0:.+]] = llvm.mlir.constant(0 : i32) : i32

  // CHECK: rocdl.mfma.f32.32x32x16.f16{{.*}}: (vector<8xf16>, vector<8xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg0 * %arg0 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf16>, vector<8xf16>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.16x16x32.f16{{.*}}: (vector<8xf16>, vector<8xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg0 * %arg0 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf16>, vector<8xf16>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x16.bf16{{.*}}: (vector<8xi16>, vector<8xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg3 * %arg3 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<8xbf16>, vector<8xbf16>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.16x16x32.bf16{{.*}}: (vector<8xi16>, vector<8xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg3 * %arg3 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
  // CHECK: rocdl.mfma.i32.32x32x32.i8{{.*}}: (vector<4xi32>, vector<4xi32>, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  amdgpu.mfma %arg4 * %arg4 + %arg5 { abid = 0 : i32, cbsz = 0 : i32, k = 32 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<16xi8>, vector<16xi8>, vector<16xi32>
  // CHECK: rocdl.mfma.i32.16x16x64.i8{{.*}}: (vector<4xi32>, vector<4xi32>, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  amdgpu.mfma %arg4 * %arg4 + %arg6 { abid = 0 : i32, cbsz = 0 : i32, k = 64 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<16xi8>, vector<16xi8>, vector<4xi32>
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4{{.*}}, %[[c0]], %[[c0]], %[[c0]], %[[c0]]{{.*}}: (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg7 * %arg7 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 64 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 } blgp = none : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<16xf32>
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4{{.*}}, %[[c0]], %[[c0]], %[[c0]], %[[c0]]{{.*}}: (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg7 * %arg7 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 128 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 } blgp = none : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<4xf32>
  // CHECK: %[[c1:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4{{.*}}, %[[c1]], %[[c1]], %[[c0]], %[[c0]]{{.*}}: (vector<8xi32>, vector<8xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg8 * %arg8 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 64 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 } blgp = none : vector<32xf8E5M2>, vector<32xf8E5M2>, vector<16xf32>
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4{{.*}}, %[[c1]], %[[c1]], %[[c0]], %[[c0]]{{.*}}: (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg8 * %arg8 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 128 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 } blgp = none : vector<32xf8E5M2>, vector<32xf8E5M2>, vector<4xf32>
  // CHECK: %[[c2:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4{{.*}}, %[[c2]], %[[c2]], %[[c0]], %[[c0]]{{.*}}: (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg9 * %arg9 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 64 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 } blgp = none : vector<32xf6E2M3FN>, vector<32xf6E2M3FN>, vector<16xf32>
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4{{.*}}, %[[c2]], %[[c2]], %[[c0]], %[[c0]]{{.*}}: (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg9 * %arg9 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 128 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 } blgp = none : vector<32xf6E2M3FN>, vector<32xf6E2M3FN>, vector<4xf32>
  // CHECK: %[[c3:.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4{{.*}}, %[[c3]], %[[c3]], %[[c0]], %[[c0]]{{.*}}: (vector<6xi32>, vector<6xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg10 * %arg10 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 64 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 } blgp = none : vector<32xf6E3M2FN>, vector<32xf6E3M2FN>, vector<16xf32>
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4{{.*}}, %[[c3]], %[[c3]], %[[c0]], %[[c0]]{{.*}}: (vector<6xi32>, vector<6xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg10 * %arg10 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 128 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 } blgp = none : vector<32xf6E3M2FN>, vector<32xf6E3M2FN>, vector<4xf32>
  // CHECK-DAG: %[[c4:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4{{.*}}, %[[c4]], %[[c4]], %[[c0]], %[[c0]]{{.*}}: (vector<4xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg11 * %arg11 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 64 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 } blgp = none : vector<32xf4E2M1FN>, vector<32xf4E2M1FN>, vector<16xf32>
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4{{.*}}, %[[c4]], %[[c4]], %[[c0]], %[[c0]]{{.*}}: (vector<4xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg11 * %arg11 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 128 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 } blgp = none : vector<32xf4E2M1FN>, vector<32xf4E2M1FN>, vector<4xf32>

  // CHECK: rocdl.mfma.scale.f32.32x32x64.f8f6f4{{.*}}, %[[c2]], %[[c4]], %[[c0]], %[[c0]]{{.*}}: (vector<6xi32>, vector<4xi32>, vector<16xf32>, i32, i32, i32, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg9 * %arg11 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 64 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 } blgp = none : vector<32xf6E2M3FN>, vector<32xf4E2M1FN>, vector<16xf32>
  // CHECK: rocdl.mfma.scale.f32.16x16x128.f8f6f4{{.*}}, %[[c2]], %[[c4]], %[[c0]], %[[c0]]{{.*}}: (vector<6xi32>, vector<4xi32>, vector<4xf32>, i32, i32, i32, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg9 * %arg11 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 128 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 } blgp = none : vector<32xf6E2M3FN>, vector<32xf4E2M1FN>, vector<4xf32>

  func.return
}
