// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx950 -cse | FileCheck %s
func.func @sparse_mfma_to_rocdl(%arg0 : vector<8xf16>, %arg1 : vector<16xf16>,
                                %arg2 : vector<4xf32>, %arg3 : vector<16xf32>,
                                %arg4 : vector<8xbf16>, %arg5 : vector<16xbf16>,
                                %arg6 : vector<16xi8>, %arg7 : vector<32xi8>,
                                %arg8 : vector<4xi32>, %arg9 : vector<16xi32>,
                                %arg10 : vector<16xf8E4M3FN>, %arg11 : vector<16xf8E5M2>,
                                %arg12 : vector<32xf8E4M3FN>, %arg13 : vector<32xf8E5M2>,
                                %arg14 : vector<4xi8>, %arg15 : vector<2xi16>) {
  // CHECK: llvm.bitcast %{{.*}} : vector<4xi8> to i32
  // CHECK: rocdl.smfmac.f32.16x16x64.f16{{.*}}: (vector<8xf16>, vector<16xf16>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x64 %arg0 * %arg1 + %arg2 sparse(%arg14 : vector<4xi8>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf16>, vector<16xf16>, vector<4xf32>

  // CHECK: rocdl.smfmac.f32.16x16x64.bf16{{.*}}: (vector<8xbf16>, vector<16xbf16>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x64 %arg4 * %arg5 + %arg2 sparse(%arg14 : vector<4xi8>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xbf16>, vector<16xbf16>, vector<4xf32>

  // CHECK: llvm.bitcast {{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<32xi8> to vector<8xi32>
  // CHECK: llvm.bitcast %{{.*}} : vector<2xi16> to i32
  // CHECK: rocdl.smfmac.i32.16x16x128.i8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<4xi32>, i32) -> vector<4xi32>
  amdgpu.sparse_mfma 16x16x128 %arg6 * %arg7 + %arg8 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xi8>, vector<32xi8>, vector<4xi32>

  // CHECK: llvm.bitcast {{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<32xi8> to vector<8xi32>
  // CHECK: rocdl.smfmac.f32.16x16x128.fp8.fp8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x128 %arg10 * %arg12 + %arg2 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xf8E4M3FN>, vector<32xf8E4M3FN>, vector<4xf32>

  // CHECK: llvm.bitcast {{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<32xi8> to vector<8xi32>
  // CHECK: rocdl.smfmac.f32.16x16x128.bf8.bf8 {{.*}}: (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x128 %arg11 * %arg13 + %arg2 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xf8E5M2>, vector<32xf8E5M2>, vector<4xf32>

  // CHECK: rocdl.smfmac.f32.16x16x128.fp8.bf8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x128 %arg10 * %arg13 + %arg2 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xf8E4M3FN>, vector<32xf8E5M2>, vector<4xf32>

  // CHECK: rocdl.smfmac.f32.16x16x128.bf8.fp8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x128 %arg11 * %arg12 + %arg2 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xf8E5M2>, vector<32xf8E4M3FN>, vector<4xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.f16{{.*}}: (vector<8xf16>, vector<16xf16>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x32 %arg0 * %arg1 + %arg3 sparse(%arg14 : vector<4xi8>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf16>, vector<16xf16>, vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.bf16{{.*}}: (vector<8xbf16>, vector<16xbf16>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x32 %arg4 * %arg5 + %arg3 sparse(%arg14 : vector<4xi8>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xbf16>, vector<16xbf16>, vector<16xf32>

  // CHECK: rocdl.smfmac.i32.32x32x64.i8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<16xi32>, i32) -> vector<16xi32>
  amdgpu.sparse_mfma 32x32x64 %arg6 * %arg7 + %arg9 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xi8>, vector<32xi8>, vector<16xi32>

  // CHECK: rocdl.smfmac.f32.32x32x64.fp8.fp8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x64 %arg10 * %arg12 + %arg3 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xf8E4M3FN>, vector<32xf8E4M3FN>, vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x64.bf8.bf8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x64 %arg11 * %arg13 + %arg3 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xf8E5M2>, vector<32xf8E5M2>, vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x64.fp8.bf8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x64 %arg10 * %arg13 + %arg3 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xf8E4M3FN>, vector<32xf8E5M2>, vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x64.bf8.fp8{{.*}}: (vector<4xi32>, vector<8xi32>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x64 %arg11 * %arg12 + %arg3 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<16xf8E5M2>, vector<32xf8E4M3FN>, vector<16xf32>

  func.return
}
