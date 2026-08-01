// RUN: mlir-opt %s -convert-nvgpu-to-nvvm -split-input-file | FileCheck %s

// CHECK-LABEL: @fp8_mma_e4m3_e4m3_m16n8k16
func.func @fp8_mma_e4m3_e4m3_m16n8k16(%arg0: vector<2x4xf8E4M3FN>, %arg1: vector<1x4xf8E4M3FN>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // CHECK: nvvm.mma.sync
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<e4m3>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<e4m3>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
  %0 = nvgpu.mma.sync(%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<2x4xf8E4M3FN>, vector<1x4xf8E4M3FN>, vector<2x2xf32>) -> vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @fp8_mma_e4m3_e4m3_m16n8k32
func.func @fp8_mma_e4m3_e4m3_m16n8k32(%arg0: vector<4x4xf8E4M3FN>, %arg1: vector<2x4xf8E4M3FN>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // CHECK: nvvm.mma.sync
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<e4m3>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<e4m3>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 32>
  %0 = nvgpu.mma.sync(%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<4x4xf8E4M3FN>, vector<2x4xf8E4M3FN>, vector<2x2xf32>) -> vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @fp8_mma_e5m2_e5m2_m16n8k16
func.func @fp8_mma_e5m2_e5m2_m16n8k16(%arg0: vector<2x4xf8E5M2>, %arg1: vector<1x4xf8E5M2>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // CHECK: nvvm.mma.sync
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<e5m2>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<e5m2>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
  %0 = nvgpu.mma.sync(%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<2x4xf8E5M2>, vector<1x4xf8E5M2>, vector<2x2xf32>) -> vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @fp8_mma_e5m2_e5m2_m16n8k32
func.func @fp8_mma_e5m2_e5m2_m16n8k32(%arg0: vector<4x4xf8E5M2>, %arg1: vector<2x4xf8E5M2>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // CHECK: nvvm.mma.sync
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<e5m2>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<e5m2>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 32>
  %0 = nvgpu.mma.sync(%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<4x4xf8E5M2>, vector<2x4xf8E5M2>, vector<2x2xf32>) -> vector<2x2xf32>
  return %0 : vector<2x2xf32>
}
