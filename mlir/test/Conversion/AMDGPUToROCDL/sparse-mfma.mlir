// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx942 -cse | FileCheck %s
func.func @sparse_mfma_to_rocdl(%arg0 : vector<4xf16>, %arg1 : vector<8xf16>,
                                %arg2 : vector<4xf32>, %arg3 : vector<16xf32>,
                                %arg4 : vector<4xbf16>, %arg5 : vector<8xbf16>,
                                %arg6 : vector<8xi8>, %arg7 : vector<16xi8>,
                                %arg8 : vector<4xi32>, %arg9 : vector<16xi32>,
                                %arg10 : vector<8xf8E4M3FNUZ>, %arg11 : vector<8xf8E5M2FNUZ>,
                                %arg12 : vector<16xf8E4M3FNUZ>, %arg13 : vector<16xf8E5M2FNUZ>,
                                %arg14 : vector<4xi8>, %arg15 : vector<2xi16>) {
  // CHECK: llvm.bitcast %{{.*}} : vector<4xi8> to i32
  // CHECK: rocdl.smfmac.f32.16x16x32.f16{{.*}}: (vector<4xf16>, vector<8xf16>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x32 %arg0 * %arg1 + %arg2 sparse(%arg14 : vector<4xi8>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<4xf16>, vector<8xf16>, vector<4xf32>

  // CHECK: llvm.bitcast {{.*}} : vector<4xbf16> to vector<4xi16>
  // CHECK: llvm.bitcast {{.*}} : vector<8xbf16> to vector<8xi16>
  // CHECK: rocdl.smfmac.f32.16x16x32.bf16 {{.*}}: (vector<4xi16>, vector<8xi16>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x32 %arg4 * %arg5 + %arg2 sparse(%arg14 : vector<4xi8>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<4xbf16>, vector<8xbf16>, vector<4xf32>

  // CHECK: rocdl.smfmac.f32.32x32x16.f16{{.*}}: (vector<4xf16>, vector<8xf16>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x16 %arg0 * %arg1 + %arg3 sparse(%arg14 : vector<4xi8>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<4xf16>, vector<8xf16>, vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x16.bf16 {{.*}}: (vector<4xi16>, vector<8xi16>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x16 %arg4 * %arg5 + %arg3 sparse(%arg14 : vector<4xi8>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<4xbf16>, vector<8xbf16>, vector<16xf32>

  // CHECK: llvm.bitcast {{.*}} : vector<8xi8> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: llvm.bitcast %{{.*}} : vector<2xi16> to i32
  // CHECK: rocdl.smfmac.i32.16x16x64.i8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<4xi32>, i32) -> vector<4xi32>
  amdgpu.sparse_mfma 16x16x64 %arg6 * %arg7 + %arg8 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xi8>, vector<16xi8>, vector<4xi32>

  // CHECK: llvm.bitcast {{.*}} : vector<8xi8> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: rocdl.smfmac.f32.16x16x64.fp8.fp8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x64 %arg10 * %arg12 + %arg2 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf8E4M3FNUZ>, vector<16xf8E4M3FNUZ>, vector<4xf32>

  // CHECK: llvm.bitcast {{.*}} : vector<8xi8> to vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: rocdl.smfmac.f32.16x16x64.bf8.bf8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x64 %arg11 * %arg13 + %arg2 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf8E5M2FNUZ>, vector<16xf8E5M2FNUZ>, vector<4xf32>

  // CHECK: rocdl.smfmac.f32.16x16x64.fp8.bf8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x64 %arg10 * %arg13 + %arg2 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf8E4M3FNUZ>, vector<16xf8E5M2FNUZ>, vector<4xf32>

  // CHECK: rocdl.smfmac.f32.16x16x64.bf8.fp8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<4xf32>, i32) -> vector<4xf32>
  amdgpu.sparse_mfma 16x16x64 %arg11 * %arg12 + %arg2 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf8E5M2FNUZ>, vector<16xf8E4M3FNUZ>, vector<4xf32>

  // CHECK: rocdl.smfmac.i32.32x32x32.i8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<16xi32>, i32) -> vector<16xi32>
  amdgpu.sparse_mfma 32x32x32 %arg6 * %arg7 + %arg9 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xi8>, vector<16xi8>, vector<16xi32>

  // CHECK: rocdl.smfmac.f32.32x32x32.fp8.fp8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x32 %arg10 * %arg12 + %arg3 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf8E4M3FNUZ>, vector<16xf8E4M3FNUZ>, vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.bf8.bf8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x32 %arg11 * %arg13 + %arg3 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf8E5M2FNUZ>, vector<16xf8E5M2FNUZ>, vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.fp8.bf8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x32 %arg10 * %arg13 + %arg3 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf8E4M3FNUZ>, vector<16xf8E5M2FNUZ>, vector<16xf32>

  // CHECK: rocdl.smfmac.f32.32x32x32.bf8.fp8{{.*}}: (vector<2xi32>, vector<4xi32>, vector<16xf32>, i32) -> vector<16xf32>
  amdgpu.sparse_mfma 32x32x32 %arg11 * %arg12 + %arg3 sparse(%arg15 : vector<2xi16>) { abid = 0 : i32, cbsz = 0 : i32 } : vector<8xf8E5M2FNUZ>, vector<16xf8E4M3FNUZ>, vector<16xf32>

  func.return
}
