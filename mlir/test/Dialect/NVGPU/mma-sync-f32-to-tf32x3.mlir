// RUN: mlir-opt %s -test-nvgpu-mmasync-f32-to-tf32-patterns="precision=tf32x3" -split-input-file | FileCheck %s

// CHECK-LABEL: m16n8k4_tf32
func.func @m16n8k4_tf32(%arg0: vector<2x1xf32>, %arg1: vector<1x1xf32>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {  
  // expected-error @+1 {{TF32x3 is not supported at the moment for nvgpu.mma.sync on f32 datatype}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 4]} : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>  
  return %d : vector<2x2xf32>
}

// -----

// CHECK-LABEL: m16n8k8_tf32
func.func @m16n8k8_tf32(%arg0: vector<4x1xf32>, %arg1: vector<2x1xf32>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // expected-error @+1 {{TF32x3 is not supported at the moment for nvgpu.mma.sync on f32 datatype}}
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 8]} : (vector<4x1xf32>, vector<2x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>    
  return %d : vector<2x2xf32>
}
// -----
