// RUN: mlir-opt %s -convert-vector-to-rocdl | FileCheck %s

gpu.module @test_read{
func.func @transfer_readx2(%A : memref<?xf32>, %base: index) -> vector<2xf32> {
  %f0 = arith.constant 0.0: f32
  %f = vector.transfer_read %A[%base], %f0
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<2xf32>
  return %f: vector<2xf32>
}
// CHECK-LABEL: @transfer_readx2
// CHECK: rocdl.buffer.load {{.*}} vector<2xf32>

func.func @transfer_readx4(%A : memref<?xf32>, %base: index) -> vector<4xf32> {
  %f0 = arith.constant 0.0: f32
  %f = vector.transfer_read %A[%base], %f0
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<4xf32>
  return %f: vector<4xf32>
}
// CHECK-LABEL: @transfer_readx4
// CHECK: rocdl.buffer.load {{.*}} vector<4xf32>

func.func @transfer_read_dwordConfig(%A : memref<?xf32>, %base: index) -> vector<4xf32> {
  %f0 = arith.constant 0.0: f32
  %f = vector.transfer_read %A[%base], %f0
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<4xf32>
  return %f: vector<4xf32>
}
// CHECK-LABEL: @transfer_read_dwordConfig
// CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}}
// CHECK: [0, 0, -1, 159744]
// CHECK: %[[i64:.*]] = llvm.ptrtoint %[[gep]]
// CHECK: llvm.insertelement %[[i64]]
}

gpu.module @test_write{
func.func @transfer_writex2(%A : memref<?xf32>, %B : vector<2xf32>, %base: index) {
  vector.transfer_write %B, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<2xf32>, memref<?xf32>
  return
}
// CHECK-LABEL: @transfer_writex2
// CHECK: rocdl.buffer.store {{.*}} vector<2xf32>

func.func @transfer_writex4(%A : memref<?xf32>, %B : vector<4xf32>, %base: index) {
  vector.transfer_write %B, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<4xf32>, memref<?xf32>
  return
}
// CHECK-LABEL: @transfer_writex4
// CHECK: rocdl.buffer.store {{.*}} vector<4xf32>

func.func @transfer_write_dwordConfig(%A : memref<?xf32>, %B : vector<2xf32>, %base: index) {
  vector.transfer_write %B, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<2xf32>, memref<?xf32>
  return
}
// CHECK-LABEL: @transfer_write_dwordConfig
// CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}}
// CHECK: [0, 0, -1, 159744]
// CHECK: %[[i64:.*]] = llvm.ptrtoint %[[gep]]
// CHECK: llvm.insertelement %[[i64]]
}
