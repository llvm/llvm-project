// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx1250 \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @wmma_k4
func.func @wmma_k4(%arg0 : vector<2xf32>, %arg1 : vector<8xf32>) {
  // CHECK: rocdl.wmma.f32.16x16x4.f32 %arg0, %arg0, %arg1
  amdgpu.wmma 16x16x4 %arg0 * %arg0 + %arg1 : vector<2xf32>, vector<2xf32>, vector<8xf32>
  return
}

// CHECK-LABEL: @wmma_k32
func.func @wmma_k32(%arg0 : vector<16xf16>, %arg1 : vector<16xbf16>, %arg2 : vector<8xf32>,
                    %arg3 : vector<8xf16>, %arg4 : vector<8xbf16>) {
  // CHECK: rocdl.wmma.f32.16x16x32.f16 %arg0, %arg0, %arg2
  amdgpu.wmma 16x16x32 %arg0 * %arg0 + %arg2 : vector<16xf16>, vector<16xf16>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x32.f16 %arg0, %arg0, {{.*}} : (vector<16xf16>, vector<16xf16>, vector<8xf16>)
  amdgpu.wmma 16x16x32 %arg0 * %arg0 + %arg3 : vector<16xf16>, vector<16xf16>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x32.bf16 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x32 %arg1 * %arg1 + %arg2 : vector<16xbf16>, vector<16xbf16>, vector<8xf32>

  // CHECK: rocdl.wmma.bf16.16x16x32.bf16 {{.*}}, {{.*}}, {{.*}} : (vector<16xbf16>, vector<16xbf16>, vector<8xbf16>)
  amdgpu.wmma 16x16x32 %arg1 * %arg1 + %arg4 : vector<16xbf16>, vector<16xbf16>, vector<8xbf16>

  return
}

// CHECK-LABEL: @wmma_k64
func.func @wmma_k64(%arg0 : vector<32xi8>, %arg1 : vector<32xf8E4M3FN>, %arg2 : vector<32xf8E5M2>,
                    %arg3 : vector<8xi32>, %arg4 : vector<8xf32>, %arg5 : vector<8xf16>) {
  // CHECK: rocdl.wmma.i32.16x16x64.iu8 {{.*}}, {{.*}}, %arg3 {clamp = true, signA = true, signB = true}
  amdgpu.wmma 16x16x64 %arg0 * %arg0 + %arg3 {clamp} : vector<32xi8>, vector<32xi8>, vector<8xi32>

  // CHECK: rocdl.wmma.f32.16x16x64.fp8_fp8 {{.*}}, {{.*}}, %arg4
  amdgpu.wmma 16x16x64 %arg1 * %arg1 + %arg4 : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x64.fp8_fp8 {{.*}}, {{.*}}, %arg5 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x64 %arg1 * %arg1 + %arg5 : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x64.fp8_bf8 {{.*}}, {{.*}}, %arg4
  amdgpu.wmma 16x16x64 %arg1 * %arg2 + %arg4 : vector<32xf8E4M3FN>, vector<32xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x64.fp8_bf8 {{.*}}, {{.*}}, %arg5 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x64 %arg1 * %arg2 + %arg5 : vector<32xf8E4M3FN>, vector<32xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x64.bf8_bf8 {{.*}}, {{.*}}, %arg4
  amdgpu.wmma 16x16x64 %arg2 * %arg2 + %arg4 : vector<32xf8E5M2>, vector<32xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x64.bf8_bf8 {{.*}}, {{.*}}, %arg5 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x64 %arg2 * %arg2 + %arg5 : vector<32xf8E5M2>, vector<32xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x64.bf8_fp8 {{.*}}, {{.*}}, %arg4
  amdgpu.wmma 16x16x64 %arg2 * %arg1 + %arg4 : vector<32xf8E5M2>, vector<32xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x64.bf8_fp8 {{.*}}, {{.*}}, %arg5 {{.*}} : (vector<8xi32>, vector<8xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x64 %arg2 * %arg1 + %arg5 : vector<32xf8E5M2>, vector<32xf8E4M3FN>, vector<8xf16>

  return
}

// CHECK-LABEL: @wmma_k128
func.func @wmma_k128(%arg0 : vector<64xf8E4M3FN>, %arg1 : vector<64xf8E5M2>,
                     %arg2 : vector<8xf32>, %arg3 : vector<8xf16>) {
  // CHECK: rocdl.wmma.f32.16x16x128.fp8_fp8 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x128 %arg0 * %arg0 + %arg2 : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x128.fp8_fp8 {{.*}}, {{.*}}, %arg3 {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x128 %arg0 * %arg0 + %arg3 : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x128.fp8_bf8 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x128 %arg0 * %arg1 + %arg2 : vector<64xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x128.fp8_bf8 {{.*}}, {{.*}}, %arg3 {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x128 %arg0 * %arg1 + %arg3 : vector<64xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x128.bf8_bf8 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x128 %arg1 * %arg1 + %arg2 : vector<64xf8E5M2>, vector<64xf8E5M2>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x128.bf8_bf8 {{.*}}, {{.*}}, %arg3 {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x128 %arg1 * %arg1 + %arg3 : vector<64xf8E5M2>, vector<64xf8E5M2>, vector<8xf16>

  // CHECK: rocdl.wmma.f32.16x16x128.bf8_fp8 {{.*}}, {{.*}}, %arg2
  amdgpu.wmma 16x16x128 %arg1 * %arg0 + %arg2 : vector<64xf8E5M2>, vector<64xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.f16.16x16x128.bf8_fp8 {{.*}}, {{.*}}, %arg3 {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf16>)
  amdgpu.wmma 16x16x128 %arg1 * %arg0 + %arg3 : vector<64xf8E5M2>, vector<64xf8E4M3FN>, vector<8xf16>

  return
}

// CHECK-LABEL: @wmma_scale_16x16x128_fp8
func.func @wmma_scale_16x16x128_fp8(%arg0 : vector<64xf8E4M3FN>, %arg1 : vector<64xf6E2M3FN>,
                                    %arg2 : vector<8xf32>, %arg3 : vector<4xf8E8M0FNU>) {
  // CHECK: rocdl.wmma.scale.f32.16x16x128.f8f6f4 {{.*}}, {{.*}}, %arg2, {{.*}}, {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>
  %0 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg0) * (%arg3 * %arg0) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.scale.f32.16x16x128.f8f6f4 {{.*}}, {{.*}}, %arg2, {{.*}}, {{.*}} {fmtA = 2 : i32, fmtB = 2 : i32, scaleAType = 1 : i32} : (vector<12xi32>, vector<12xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>
  %1 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg1) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 16 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf6E2M3FN>, vector<4xf8E8M0FNU>, vector<64xf6E2M3FN>, vector<8xf32>

  func.return
}

// CHECK-LABEL: @wmma_scale_16x16x128_fp6
func.func @wmma_scale_16x16x128_fp6(%arg0 : vector<64xf6E2M3FN>, %arg1 : vector<64xf6E3M2FN>,
                                    %arg2 : vector<8xf32>, %arg3 : vector<4xf8E8M0FNU>) {
  // CHECK: rocdl.wmma.scale.f32.16x16x128.f8f6f4 {{.*}}, {{.*}}, %arg2, {{.*}}, {{.*}} {fmtA = 2 : i32, fmtB = 2 : i32} : (vector<12xi32>, vector<12xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>
  %0 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg0) * (%arg3 * %arg0) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf6E2M3FN>, vector<4xf8E8M0FNU>, vector<64xf6E2M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.scale.f32.16x16x128.f8f6f4 {{.*}}, {{.*}}, %arg2, {{.*}}, {{.*}} {fmtA = 3 : i32, fmtB = 3 : i32} : (vector<12xi32>, vector<12xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>
  %1 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg1) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf6E3M2FN>, vector<4xf8E8M0FNU>, vector<64xf6E3M2FN>, vector<8xf32>

  func.return
}

// CHECK-LABEL: @wmma_scale_16x16x128_mixed
func.func @wmma_scale_16x16x128_mixed(%arg0 : vector<64xf8E4M3FN>, %arg1 : vector<64xf6E2M3FN>,
                                      %arg2 : vector<64xf4E2M1FN>, %arg3 : vector<8xf32>,
                                      %arg4 : vector<4xf8E8M0FNU>, %arg5 : vector<4xf8E4M3FN>) {
  // CHECK: rocdl.wmma.scale.f32.16x16x128.f8f6f4 {{.*}}, {{.*}}, %arg3, {{.*}}, {{.*}} {fmtB = 4 : i32, fmtScaleB = 2 : i32} : (vector<16xi32>, vector<8xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>
  %0 = amdgpu.scaled_wmma 16x16x128 (%arg4 * %arg0) * (%arg5 * %arg2) + %arg3 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E4M3FN>, vector<64xf4E2M1FN>, vector<8xf32>

  // CHECK: rocdl.wmma.scale.f32.16x16x128.f8f6f4 {{.*}}, {{.*}}, %arg3, {{.*}}, {{.*}} {fmtA = 2 : i32, fmtB = 4 : i32, fmtScaleB = 2 : i32} : (vector<12xi32>, vector<8xi32>, vector<8xf32>, i32, i32) -> vector<8xf32>
  %1 = amdgpu.scaled_wmma 16x16x128 (%arg4 * %arg1) * (%arg5 * %arg2) + %arg3 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf6E2M3FN>, vector<4xf8E4M3FN>, vector<64xf4E2M1FN>, vector<8xf32>

  func.return
}

// CHECK-LABEL: @wmma_scale16_16x16x128_fp8
func.func @wmma_scale16_16x16x128_fp8(%arg0 : vector<64xf8E4M3FN>, %arg1 : vector<64xf6E3M2FN>,
                                      %arg2 : vector<8xf32>, %arg3 : vector<8xf8E8M0FNU>) {
  // CHECK: rocdl.wmma.scale16.f32.16x16x128.f8f6f4 {{.*}}, {{.*}}, %arg2, {{.*}}, {{.*}} : (vector<16xi32>, vector<16xi32>, vector<8xf32>, i64, i64) -> vector<8xf32>
  %0 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg0) * (%arg3 * %arg0) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<8xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<8xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<8xf32>

  // CHECK: rocdl.wmma.scale16.f32.16x16x128.f8f6f4 {{.*}}, {{.*}}, %arg2, {{.*}}, {{.*}} {fmtA = 3 : i32, fmtB = 3 : i32, scaleAType = 1 : i32} : (vector<12xi32>, vector<12xi32>, vector<8xf32>, i64, i64) -> vector<8xf32>
  %1 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg1) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 16 : i32, b_first_scale_lane = 0 : i32} : vector<8xf8E8M0FNU>, vector<64xf6E3M2FN>, vector<8xf8E8M0FNU>, vector<64xf6E3M2FN>, vector<8xf32>

  func.return
}

// CHECK-LABEL: @wmma_scale_32x16x128_fp4
func.func @wmma_scale_32x16x128_fp4(%arg0 : vector<128xf4E2M1FN>, %arg1 : vector<64xf4E2M1FN>,
                                    %arg2 : vector<16xf32>, %arg3 : vector<4xf8E4M3FN>) {
  // CHECK: rocdl.wmma.scale.f32.32x16x128.f4 {{.*}}, {{.*}}, %arg2, {{.*}}, {{.*}} {fmtScaleA = 2 : i32, fmtScaleB = 2 : i32} : (vector<16xi32>, vector<8xi32>, vector<16xf32>, i32, i32) -> vector<16xf32>
  %0 = amdgpu.scaled_wmma 32x16x128 (%arg3 * %arg0) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E4M3FN>, vector<128xf4E2M1FN>, vector<4xf8E4M3FN>, vector<64xf4E2M1FN>, vector<16xf32>

  func.return
}

// CHECK-LABEL: @wmma_scale16_32x16x128_fp4
func.func @wmma_scale16_32x16x128_fp4(%arg0 : vector<128xf4E2M1FN>, %arg1 : vector<64xf4E2M1FN>,
                                      %arg2 : vector<16xf32>, %arg3 : vector<8xf8E4M3FN>) {
  // CHECK: rocdl.wmma.scale16.f32.32x16x128.f4 {{.*}}, {{.*}}, %arg2, {{.*}}, {{.*}} {fmtScaleA = 2 : i32, fmtScaleB = 2 : i32} : (vector<16xi32>, vector<8xi32>, vector<16xf32>, i64, i64) -> vector<16xf32>
  %0 = amdgpu.scaled_wmma 32x16x128 (%arg3 * %arg0) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<8xf8E4M3FN>, vector<128xf4E2M1FN>, vector<8xf8E4M3FN>, vector<64xf4E2M1FN>, vector<16xf32>

  func.return
}

// -----

func.func @wmma_unsupported_k(%arg0 : vector<8xf16>, %arg1 : vector<8xf32>) {
  // expected-error@below {{'amdgpu.wmma' op no intrinsic matching WMMA on the given chipset}}
  // expected-error@below {{failed to legalize operation 'amdgpu.wmma'}}
  amdgpu.wmma 16x16x16 %arg0 * %arg0 + %arg1 : vector<8xf16>, vector<8xf16>, vector<8xf32>
  return
}

// -----

func.func @scaled_wmma_wrong_output_length(%arg0 : vector<64xf8E4M3FN>, %arg1 : vector<16xf32>,
                                           %arg2 : vector<4xf8E8M0FNU>) {
  // expected-error@below {{'amdgpu.scaled_wmma' op expected output vector of length 8 but got 16}}
  %0 = amdgpu.scaled_wmma 16x16x128 (%arg2 * %arg0) * (%arg2 * %arg0) + %arg1 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<16xf32>
  return
}

func.func @scaled_wmma_16x16_wrong_sourceA_length(%arg0 : vector<128xf4E2M1FN>, %arg1 : vector<64xf4E2M1FN>,
                                                  %arg2 : vector<8xf32>, %arg3 : vector<4xf8E8M0FNU>) {
  // expected-error@below {{'amdgpu.scaled_wmma' op for 16x16x128, sourceA must have 64 elements but got 128}}
  %0 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg0) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<128xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<64xf4E2M1FN>, vector<8xf32>
  return
}

func.func @scaled_wmma_16x16_wrong_sourceB_length(%arg0 : vector<64xf8E4M3FN>, %arg1 : vector<128xf4E2M1FN>,
                                                  %arg2 : vector<8xf32>, %arg3 : vector<4xf8E8M0FNU>) {
  // expected-error@below {{'amdgpu.scaled_wmma' op for 16x16x128, sourceB must have 64 elements but got 128}}
  %0 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg0) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E8M0FNU>, vector<128xf4E2M1FN>, vector<8xf32>
  return
}

func.func @scaled_wmma_32x16_wrong_sourceA_length(%arg0 : vector<64xf4E2M1FN>, %arg1 : vector<64xf4E2M1FN>,
                                                  %arg2 : vector<16xf32>, %arg3 : vector<4xf8E4M3FN>) {
  // expected-error@below {{'amdgpu.scaled_wmma' op for 32x16x128, sourceA must have 128 elements but got 64}}
  %0 = amdgpu.scaled_wmma 32x16x128 (%arg3 * %arg0) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E4M3FN>, vector<64xf4E2M1FN>, vector<4xf8E4M3FN>, vector<64xf4E2M1FN>, vector<16xf32>
  return
}

func.func @scaled_wmma_32x16_wrong_sourceB_length(%arg0 : vector<128xf4E2M1FN>, %arg1 : vector<128xf4E2M1FN>,
                                                  %arg2 : vector<16xf32>, %arg3 : vector<4xf8E4M3FN>) {
  // expected-error@below {{'amdgpu.scaled_wmma' op for 32x16x128, sourceB must have 64 elements but got 128}}
  %0 = amdgpu.scaled_wmma 32x16x128 (%arg3 * %arg0) * (%arg3 * %arg1) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E4M3FN>, vector<128xf4E2M1FN>, vector<4xf8E4M3FN>, vector<128xf4E2M1FN>, vector<16xf32>
  return
}

func.func @scaled_wmma_invalid_type_combination(%arg0 : vector<64xf8E4M3FN>, %arg1 : vector<64xf6E2M3FN>,
                                                %arg2 : vector<8xf32>, %arg3 : vector<4xf8E8M0FNU>,
                                                %arg4 : vector<4xf8E4M3FN>) {
  // expected-error@below {{'amdgpu.scaled_wmma' op invalid combination of matrix and scale types}}
  %0 = amdgpu.scaled_wmma 16x16x128 (%arg3 * %arg0) * (%arg4 * %arg1) + %arg2 {a_first_scale_lane = 0 : i32, b_first_scale_lane = 0 : i32} : vector<4xf8E8M0FNU>, vector<64xf8E4M3FN>, vector<4xf8E4M3FN>, vector<64xf6E2M3FN>, vector<8xf32>
  return
}
