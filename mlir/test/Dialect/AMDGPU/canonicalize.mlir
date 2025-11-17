// RUN: mlir-opt %s -split-input-file -canonicalize="test-convergence" | FileCheck %s

// CHECK-LABEL: func @known_oob_load
func.func @known_oob_load(%arg0: memref<4xf32>) -> f32 {
  // CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: return %[[zero]]
  %c4_i32 = arith.constant 4 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %arg0[%c4_i32] : memref<4xf32>, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @known_oob_load_2d
func.func @known_oob_load_2d(%arg0: memref<4x4xf32>) -> f32 {
  // CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: return %[[zero]]
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %arg0[%c4_i32, %c0_i32] : memref<4x4xf32>, i32, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @known_oob_load_2d_on_last
func.func @known_oob_load_2d_on_last(%arg0: memref<4x4xf32>) -> f32 {
  // CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: return %[[zero]]
  %c0_i32 = arith.constant 0 : i32
  %c16_i32 = arith.constant 16 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %arg0[%c0_i32, %c16_i32] : memref<4x4xf32>, i32, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @known_oob_load_index
func.func @known_oob_load_index(%arg0: memref<4xf32>) -> f32 {
  // CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: return %[[zero]]
  %c0_i32 = arith.constant 0 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 4 : i32} %arg0[%c0_i32] : memref<4xf32>, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @known_oob_load_sgproffset
func.func @known_oob_load_sgproffset(%arg0: memref<4xf32>) -> f32 {
  // CHECK: %[[zero:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: return %[[zero]]
  %c2_i32 = arith.constant 2 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %arg0[%c2_i32] sgprOffset %c2_i32 : memref<4xf32>, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @unknown_load
func.func @unknown_load(%arg0: memref<4xf32>, %arg1: i32) -> f32 {
  // CHECK: %[[loaded:.*]] = amdgpu.raw_buffer_load
  // CHECK: return %[[loaded]]
  %c4_i32 = arith.constant 4 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %arg0[%arg1] sgprOffset %c4_i32 : memref<4xf32>, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @unknown_load_sgproffset
func.func @unknown_load_sgproffset(%arg0: memref<4xf32>, %arg1: i32) -> f32 {
  // CHECK: %[[loaded:.*]] = amdgpu.raw_buffer_load
  // CHECK: return %[[loaded]]
  %c4_i32 = arith.constant 4 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %arg0[%c4_i32] sgprOffset %arg1 : memref<4xf32>, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @unranked
func.func @unranked(%arg0: memref<?xf32>) -> f32 {
  // CHECK: %[[loaded:.*]] = amdgpu.raw_buffer_load
  // CHECK: return %[[loaded]]
  %c4_i32 = arith.constant 4 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %arg0[%c4_i32] : memref<?xf32>, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @no_oob_check
func.func @no_oob_check(%arg0: memref<4xf32>) -> f32 {
  // CHECK: %[[loaded:.*]] = amdgpu.raw_buffer_load
  // CHECK: return %[[loaded]]
  %c4_i32 = arith.constant 4 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = false} %arg0[%c4_i32] : memref<4xf32>, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @in_bounds_overall
func.func @in_bounds_overall(%arg0: memref<4x4xf32>) -> f32 {
  // CHECK: %[[loaded:.*]] = amdgpu.raw_buffer_load
  // CHECK: return %[[loaded]]
  %c0_i32 = arith.constant 0 : i32
  %c15_i32 = arith.constant 15 : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %arg0[%c0_i32, %c15_i32] : memref<4x4xf32>, i32, i32 -> f32
  func.return %0 : f32
}

// -----

// CHECK-LABEL: func @dead_store
func.func @dead_store(%arg0: memref<4xf32>, %arg1: f32) {
  // CHECK-NOT: amdgpu.raw_buffer_store
  %c4_i32 = arith.constant 4 : i32
  amdgpu.raw_buffer_store {boundsCheck = true} %arg1 -> %arg0[%c4_i32] : f32 -> memref<4xf32>, i32
  func.return
}

// -----

// CHECK-LABEL: func @dead_atomic_add
func.func @dead_atomic_add(%arg0: memref<4xf32>, %arg1: f32) {
  // CHECK-NOT: amdgpu.raw_buffer_atomic_fadd
  %c4_i32 = arith.constant 4 : i32
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true} %arg1 -> %arg0[%c4_i32] : f32 -> memref<4xf32>, i32
  func.return
}

// -----

// CHECK-LABEL: func @fold_gather_to_lds_of_cast
func.func @fold_gather_to_lds_of_cast(%global: memref<128x72xf32, 1>, %lds: memref<64x64xf32, 3>) {
// CHECK-SAME: %[[GLOBAL:[A-Za-z0-9]+]]: memref<128x72xf32, 1>
  %c0 = arith.constant 0 : index
  %0 = memref.cast %global : memref<128x72xf32, 1> to memref<?x?xf32, 1>
  // CHECK: amdgpu.gather_to_lds %[[GLOBAL]]
  // CHECK-SAME: : f32, memref<128x72xf32, 1>
  amdgpu.gather_to_lds %0[%c0, %c0], %lds[%c0, %c0]
    : f32, memref<?x?xf32, 1>, memref<64x64xf32, 3>
  func.return
}

// -----

// CHECK-LABEL: func @fold_gather_to_lds_of_cast_dest
func.func @fold_gather_to_lds_of_cast_dest(%global: memref<128x72xf32, 1>, %lds: memref<64x64xf32, 3>) {
// CHECK-SAME: %[[GLOBAL:[A-Za-z0-9]+]]: memref<128x72xf32, 1>
// CHECK-SAME: %[[LDS:[A-Za-z0-9]+]]: memref<64x64xf32, 3>
  %c0 = arith.constant 0 : index
  %0 = memref.cast %lds : memref<64x64xf32, 3> to memref<?x?xf32, 3>
  // CHECK: amdgpu.gather_to_lds %[[GLOBAL]][{{.*}}], %[[LDS]]
  // CHECK-SAME: : f32, memref<128x72xf32, 1>, memref<64x64xf32, 3>
  amdgpu.gather_to_lds %global[%c0, %c0], %0[%c0, %c0]
    : f32, memref<128x72xf32, 1>, memref<?x?xf32, 3>
  func.return
}

// -----

// CHECK-LABEL: func @scaled_mfma
// CHECK: %[[SCALE_1:.*]] = vector.extract_strided_slice %0 {offsets = [0], sizes = [4], strides = [1]} : vector<16xf8E8M0FNU> to vector<4xf8E8M0FNU>
// CHECK: %[[SCALE_2:.*]] = vector.extract_strided_slice %2 {offsets = [4], sizes = [4], strides = [1]} : vector<16xf8E8M0FNU> to vector<4xf8E8M0FNU>
// CHECK: amdgpu.scaled_mfma 16x16x128 (%[[SCALE_1]][3] * %{{.*}}) * (%[[SCALE_2]][2] * %{{.*}}) {{.*}}
// CHECK: %[[SCALE_3:.*]] = vector.extract_strided_slice %5 {offsets = [8], sizes = [4], strides = [1]} : vector<16xf8E8M0FNU> to vector<4xf8E8M0FNU>
// CHECK: %[[SCALE_4:.*]] = vector.extract_strided_slice %7 {offsets = [12], sizes = [4], strides = [1]} : vector<16xf8E8M0FNU> to vector<4xf8E8M0FNU>
// CHECK: amdgpu.scaled_mfma 16x16x128 (%[[SCALE_3]][1] * %{{.*}}) * (%[[SCALE_4]][0] * %{{.*}}) {{.*}}
func.func @scaled_mfma(%opA: vector<32xf4E2M1FN>, %opB: vector<32xf4E2M1FN>, %scalesA: vector<2x1x8x1xf8E8M0FNU>, %scalesB: vector<2x1x8x1xf8E8M0FNU>) -> (vector<4xf32>, vector<4xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
  %cst_1 = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
  %scaleA = vector.extract %scalesA[0, 0, 3, 0] : f8E8M0FNU from vector<2x1x8x1xf8E8M0FNU>
  %sA = vector.insert %scaleA, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %scaleB = vector.extract %scalesB[0, 0, 6, 0] : f8E8M0FNU from vector<2x1x8x1xf8E8M0FNU>
  %sB = vector.insert %scaleB, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %res_0 = amdgpu.scaled_mfma 16x16x128 (%sA[0] * %opA) * (%sB[0] * %opB) + %cst_0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %scaleC = vector.extract %scalesA[1, 0, 1, 0] : f8E8M0FNU from vector<2x1x8x1xf8E8M0FNU>
  %sC = vector.insert %scaleC, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %scaleD = vector.extract %scalesB[1, 0, 4, 0] : f8E8M0FNU from vector<2x1x8x1xf8E8M0FNU>
  %sD = vector.insert %scaleD, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %res_1 = amdgpu.scaled_mfma 16x16x128 (%sC[0] * %opA) * (%sD[0] * %opB) + %cst_0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  return %res_0, %res_1 : vector<4xf32>, vector<4xf32>
}

// -----

// CHECK-LABEL: func @scaled_mfma_less_than_4
// CHECK: vector.extract {{.*}} : f8E8M0FNU from vector<2xf8E8M0FNU>
// CHECK: vector.insert {{.*}} : f8E8M0FNU into vector<4xf8E8M0FNU>
// CHECK: vector.extract {{.*}} : f8E8M0FNU from vector<2xf8E8M0FNU>
// CHECK: vector.insert {{.*}} : f8E8M0FNU into vector<4xf8E8M0FNU>
// CHECK: amdgpu.scaled_mfma 16x16x128 ({{.*}}[0] * {{.*}}) * ({{.*}}[0] * {{.*}}
func.func @scaled_mfma_less_than_4(%opA: vector<32xf4E2M1FN>, %opB: vector<32xf4E2M1FN>, %scalesA: vector<2xf8E8M0FNU>, %scalesB: vector<2xf8E8M0FNU>) -> vector<4xf32> {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
  %cst_1 = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
  %scaleA = vector.extract %scalesA[0] : f8E8M0FNU from vector<2xf8E8M0FNU>
  %sA = vector.insert %scaleA, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %scaleB = vector.extract %scalesB[1] : f8E8M0FNU from vector<2xf8E8M0FNU>
  %sB = vector.insert %scaleB, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %res_0 = amdgpu.scaled_mfma 16x16x128 (%sA[0] * %opA) * (%sB[0] * %opB) + %cst_0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  return %res_0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @scaled_mfma_ugly_shapes
// CHECK: amdgpu.scaled_mfma 16x16x128 (%{{.*}}[0] * %{{.*}}) * (%{{.*}}[3] * %arg1) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
// CHECK: amdgpu.scaled_mfma 16x16x128 (%{{.*}}[1] * %{{.*}}) * (%{{.*}}[3] * %arg1) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
// CHECK: amdgpu.scaled_mfma 16x16x128 (%{{.*}}[2] * %{{.*}}) * (%{{.*}}[2] * %arg1) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
// CHECK: amdgpu.scaled_mfma 16x16x128 (%{{.*}}[3] * %{{.*}}) * (%{{.*}}[1] * %arg1) + %cst : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
func.func @scaled_mfma_ugly_shapes(%opA: vector<32xf4E2M1FN>, %opB: vector<32xf4E2M1FN>, %scalesA: vector<5x5xf8E8M0FNU>, %scalesB: vector<7x23xf8E8M0FNU>) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
  %cst_1 = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
  %scaleA_0_4 = vector.extract %scalesA[4, 0] : f8E8M0FNU from vector<5x5xf8E8M0FNU>
  %scaleA_0_5 = vector.extract %scalesA[4, 1] : f8E8M0FNU from vector<5x5xf8E8M0FNU>
  %scaleA_0_6 = vector.extract %scalesA[4, 2] : f8E8M0FNU from vector<5x5xf8E8M0FNU>
  %scaleA_0_7 = vector.extract %scalesA[4, 3] : f8E8M0FNU from vector<5x5xf8E8M0FNU>

  // idx = 160 => opsel = 3 (last idx of last 4 bytes)
  %scaleB_6_22 = vector.extract %scalesB[6, 22] : f8E8M0FNU from vector<7x23xf8E8M0FNU>
  // idx = 159 => opsel = 3
  %scaleB_6_21 = vector.extract %scalesB[6, 21] : f8E8M0FNU from vector<7x23xf8E8M0FNU>
  // idx = 158 => opsel = 2
  %scaleB_6_20 = vector.extract %scalesB[6, 20] : f8E8M0FNU from vector<7x23xf8E8M0FNU>
  // idx = 157 => opsel = 1
  %scaleB_6_19 = vector.extract %scalesB[6, 19] : f8E8M0FNU from vector<7x23xf8E8M0FNU>

  %sA_0_4 = vector.insert %scaleA_0_4, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %sA_0_5 = vector.insert %scaleA_0_5, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %sA_0_6 = vector.insert %scaleA_0_6, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %sA_0_7 = vector.insert %scaleA_0_7, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>

  %sB_6_22 = vector.insert %scaleB_6_22, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %sB_6_21 = vector.insert %scaleB_6_21, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %sB_6_20 = vector.insert %scaleB_6_20, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %sB_6_19 = vector.insert %scaleB_6_19, %cst_1 [0] : f8E8M0FNU into vector<4xf8E8M0FNU>

  %res_4 = amdgpu.scaled_mfma 16x16x128 (%sA_0_4[0] * %opA) * (%sB_6_22[0] * %opB) + %cst_0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %res_5 = amdgpu.scaled_mfma 16x16x128 (%sA_0_5[0] * %opA) * (%sB_6_21[0] * %opB) + %cst_0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %res_6 = amdgpu.scaled_mfma 16x16x128 (%sA_0_6[0] * %opA) * (%sB_6_20[0] * %opB) + %cst_0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  %res_7 = amdgpu.scaled_mfma 16x16x128 (%sA_0_7[0] * %opA) * (%sB_6_19[0] * %opB) + %cst_0 : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>
  return %res_4, %res_5, %res_6, %res_7 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
}
