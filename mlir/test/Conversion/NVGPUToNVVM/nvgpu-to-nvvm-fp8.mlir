// RUN: mlir-opt %s -convert-nvgpu-to-nvvm -split-input-file | FileCheck %s

// Test that FP8 (e4m3) nvgpu.mma.sync lowers to nvvm.mma.sync with the
// correct multiplicand PTX type.
func.func @fp8_mma_sync(%arg0: vector<4x4xf8E4M3FN>, %arg1: vector<2x4xf8E4M3FN>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // CHECK: nvvm.mma.sync
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<e4m3>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<e4m3>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 32>
  %0 = nvgpu.mma.sync(%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<4x4xf8E4M3FN>, vector<2x4xf8E4M3FN>, vector<2x2xf32>) -> vector<2x2xf32>
  return %0 : vector<2x2xf32>
}
