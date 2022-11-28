// RUN: mlir-opt %s -test-nvgpu-mmasync-f32-to-tf32-patterns="precision=tf32" -split-input-file | FileCheck %s

// CHECK-LABEL: m16n8k4_tf32
func.func @m16n8k4_tf32(%arg0: vector<2x1xf32>, %arg1: vector<1x1xf32>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // CHECK: nvgpu.mma.sync
  // CHECK-SAME: tf32Enabled
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 4]} : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
  return %d : vector<2x2xf32>
}

// -----

// CHECK-LABEL: m16n8k8_tf32
func.func @m16n8k8_tf32(%arg0: vector<4x1xf32>, %arg1: vector<2x1xf32>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // CHECK: nvgpu.mma.sync
  // CHECK-SAME: tf32Enabled
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 8]} : (vector<4x1xf32>, vector<2x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
  return %d : vector<2x2xf32>
}
// -----

// Negative test for non f32 case.
// CHECK-LABEL: mma_sync_f16
//   CHECK-NOT: tf32Enabled
//       CHECK: return
func.func @mma_sync_f16(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}
