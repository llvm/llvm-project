// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx940 | FileCheck %s
func.func @mfma_to_rocdl(%arg0 : f32, %arg1 : vector<32xf32>,
                    %arg2 : vector<16xf32>, %arg3 : vector<4xf32>,
                    %arg4 : vector<4xf16>, %arg5 : vector<4xi8>,
                    %arg6 : vector<32xi32>, %arg7 : vector<16xi32>,
                    %arg8 : vector<4xi32>, %arg9 : vector<2xbf16>,
                    %arg10 : vector<4xbf16>, %arg11 : f64,
                    %arg12 : vector<4xf64>, %arg13 : vector<8xi8>,
                    %arg14 : vector<2xf32>, %arg15 : vector<8xf8E5M2FNUZ>,
                    %arg16 : vector<8xf8E4M3FNUZ>) {
  // CHECK: rocdl.mfma.f32.32x32x1f32{{.*}}: (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
  amdgpu.mfma %arg0 * %arg0 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 1 : i32, m = 32 : i32, n = 32 : i32, blocks = 2 : i32 }  blgp = none : f32, f32, vector<32xf32>
  // CHECK: rocdl.mfma.f32.16x16x1f32{{.*}}: (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg0 * %arg0 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 1 : i32, m = 16 : i32, n = 16 : i32, blocks = 4 : i32 }  blgp = none : f32, f32, vector<16xf32>
  // CHECK: rocdl.mfma.f32.4x4x1f32{{.*}}: (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg0 * %arg0 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 1 : i32, m = 4 : i32, n = 4 : i32, blocks = 16 : i32 }  blgp = none : f32, f32, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x2f32{{.*}}: (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg0 * %arg0 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 2 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : f32, f32, vector<16xf32>
  // CHECK: rocdl.mfma.f32.16x16x4f32{{.*}}: (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg0 * %arg0 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : f32, f32, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x4f16{{.*}}: (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
  amdgpu.mfma %arg4 * %arg4 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 32 : i32, n = 32 : i32, blocks = 2 : i32 }  blgp = none : vector<4xf16>, vector<4xf16>, vector<32xf32>
  // CHECK: rocdl.mfma.f32.16x16x4f16{{.*}}: (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg4 * %arg4 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 16 : i32, n = 16 : i32, blocks = 4 : i32 }  blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.4x4x4f16{{.*}}: (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg4 * %arg4 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 4 : i32, n = 4 : i32, blocks = 16 : i32 }  blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x8f16{{.*}}: (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg4 * %arg4 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<4xf16>, vector<4xf16>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.16x16x16f16{{.*}}: (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg4 * %arg4 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
  // CHECK: rocdl.mfma.i32.32x32x4i8{{.*}}: (i32, i32, vector<32xi32>, i32, i32, i32) -> vector<32xi32>
  amdgpu.mfma %arg5 * %arg5 + %arg6 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 32 : i32, n = 32 : i32, blocks = 2 : i32 }  blgp = none : vector<4xi8>, vector<4xi8>, vector<32xi32>
  // CHECK: rocdl.mfma.i32.16x16x4i8{{.*}}: (i32, i32, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  amdgpu.mfma %arg5 * %arg5 + %arg7 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 16 : i32, n = 16 : i32, blocks = 4 : i32 }  blgp = none : vector<4xi8>, vector<4xi8>, vector<16xi32>
  // CHECK: rocdl.mfma.i32.4x4x4i8{{.*}}: (i32, i32, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  amdgpu.mfma %arg5 * %arg5 + %arg8 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 4 : i32, n = 4 : i32, blocks = 16 : i32 }  blgp = none : vector<4xi8>, vector<4xi8>, vector<4xi32>
  // CHECK: rocdl.mfma.i32.32x32x8i8{{.*}}: (i32, i32, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  amdgpu.mfma %arg5 * %arg5 + %arg7 { abid = 0 : i32, cbsz = 0 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<4xi8>, vector<4xi8>, vector<16xi32>
  // CHECK: rocdl.mfma.i32.16x16x16i8{{.*}}: (i32, i32, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  amdgpu.mfma %arg5 * %arg5 + %arg8 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<4xi8>, vector<4xi8>, vector<4xi32>
  // CHECK: rocdl.mfma.f32.32x32x2bf16{{.*}}: (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
  amdgpu.mfma %arg9 * %arg9 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 2 : i32, m = 32 : i32, n = 32 : i32, blocks = 2 : i32 }  blgp = none : vector<2xbf16>, vector<2xbf16>, vector<32xf32>
  // CHECK: rocdl.mfma.f32.16x16x2bf16{{.*}}: (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg9 * %arg9 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 2 : i32, m = 16 : i32, n = 16 : i32, blocks = 4 : i32 }  blgp = none : vector<2xbf16>, vector<2xbf16>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.4x4x2bf16{{.*}}: (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg9 * %arg9 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 2 : i32, m = 4 : i32, n = 4 : i32, blocks = 16 : i32 }  blgp = none : vector<2xbf16>, vector<2xbf16>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x4bf16{{.*}}: (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg9 * %arg9 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<2xbf16>, vector<2xbf16>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.16x16x8bf16{{.*}}: (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg9 * %arg9 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 8 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<2xbf16>, vector<2xbf16>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x4bf16.1k{{.*}}: (vector<4xi16>, vector<4xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
  amdgpu.mfma %arg10 * %arg10 + %arg1 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 32 : i32, n = 32 : i32, blocks = 2 : i32 }  blgp = none : vector<4xbf16>, vector<4xbf16>, vector<32xf32>
  // CHECK: rocdl.mfma.f32.16x16x4bf16.1k{{.*}}: (vector<4xi16>, vector<4xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg10 * %arg10 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 16 : i32, n = 16 : i32, blocks = 4 : i32 }  blgp = none : vector<4xbf16>, vector<4xbf16>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.4x4x4bf16.1k{{.*}}: (vector<4xi16>, vector<4xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg10 * %arg10 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 4 : i32, n = 4 : i32, blocks = 16 : i32 }  blgp = none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x8bf16.1k{{.*}}: (vector<4xi16>, vector<4xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg10 * %arg10 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<4xbf16>, vector<4xbf16>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.16x16x16bf16.1k{{.*}}: (vector<4xi16>, vector<4xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg10 * %arg10 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
  // CHECK: rocdl.mfma.f64.16x16x4f64{{.*}}: (f64, f64, vector<4xf64>, i32, i32, i32) -> vector<4xf64>
  amdgpu.mfma %arg11 * %arg11 + %arg12 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : f64, f64, vector<4xf64>
  // CHECK: rocdl.mfma.f64.4x4x4f64{{.*}}: (f64, f64, f64, i32, i32, i32) -> f64
  amdgpu.mfma %arg11 * %arg11 + %arg11 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 4 : i32, n = 4 : i32, blocks = 4 : i32 }  blgp = none : f64, f64, f64
  // CHECK: rocdl.mfma.i32.16x16x32.i8{{.*}}: (i64, i64, vector<4xi32>, i32, i32, i32) -> vector<4xi32>
  amdgpu.mfma %arg13 * %arg13 + %arg8 { abid = 0 : i32, cbsz = 0 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<8xi8>, vector<8xi8>, vector<4xi32>
  // CHECK: rocdl.mfma.i32.32x32x16.i8{{.*}}: (i64, i64, vector<16xi32>, i32, i32, i32) -> vector<16xi32>
  amdgpu.mfma %arg13 * %arg13 + %arg7 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<8xi8>, vector<8xi8>, vector<16xi32>
  // CHECK: rocdl.mfma.f32.16x16x8.xf32{{.*}}: (vector<2xf32>, vector<2xf32>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg14 * %arg14 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 8 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32, reducePrecision }  blgp = none : vector<2xf32>, vector<2xf32>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x4.xf32{{.*}}: (vector<2xf32>, vector<2xf32>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg14 * %arg14 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 4 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32, reducePrecision }  blgp = none : vector<2xf32>, vector<2xf32>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.16x16x32.bf8.bf8{{.*}}: (i64, i64, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg15 * %arg15 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf8E5M2FNUZ>, vector<8xf8E5M2FNUZ>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.16x16x32.bf8.fp8{{.*}}: (i64, i64, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg15 * %arg16 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf8E5M2FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.16x16x32.fp8.bf8{{.*}}: (i64, i64, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg16 * %arg15 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf8E4M3FNUZ>, vector<8xf8E5M2FNUZ>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.16x16x32.fp8.fp8{{.*}}: (i64, i64, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
  amdgpu.mfma %arg16 * %arg16 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>
  // CHECK: rocdl.mfma.f32.32x32x16.bf8.bf8{{.*}}: (i64, i64, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg15 * %arg15 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf8E5M2FNUZ>, vector<8xf8E5M2FNUZ>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.32x32x16.bf8.fp8{{.*}}: (i64, i64, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg15 * %arg16 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf8E5M2FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.32x32x16.fp8.bf8{{.*}}: (i64, i64, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg16 * %arg15 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf8E4M3FNUZ>, vector<8xf8E5M2FNUZ>, vector<16xf32>
  // CHECK: rocdl.mfma.f32.32x32x16.fp8.fp8{{.*}}: (i64, i64, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
  amdgpu.mfma %arg16 * %arg16 + %arg2 { abid = 0 : i32, cbsz = 0 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32, blocks = 1 : i32 }  blgp = none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>

  func.return
}
