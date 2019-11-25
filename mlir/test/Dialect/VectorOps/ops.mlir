// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @vector_transfer_ops(
func @vector_transfer_ops(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  %f0 = constant 0.0 : f32
  //
  // CHECK: %0 = vector.transfer_read
  %0 = vector.transfer_read %arg0[%c3, %c3], %f0 {permutation_map = (d0, d1)->(d0)} : memref<?x?xf32>, vector<128xf32>
  // CHECK: %1 = vector.transfer_read
  %1 = vector.transfer_read %arg0[%c3, %c3], %f0 {permutation_map = (d0, d1)->(d1, d0)} : memref<?x?xf32>, vector<3x7xf32>
  // CHECK: vector.transfer_read
  %2 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d0)} : memref<?x?xf32>,  vector<128xf32>
  // CHECK: vector.transfer_read
  %3 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d1)} : memref<?x?xf32>,  vector<128xf32>
  //
  // CHECK: vector.transfer_write
  vector.transfer_write %0, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0)} : vector<128xf32>, memref<?x?xf32>
  // CHECK: vector.transfer_write
  vector.transfer_write %1, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d1, d0)} : vector<3x7xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: extractelement
func @extractelement(%arg0: vector<4x8x16xf32>) -> (vector<8x16xf32>, vector<16xf32>, f32) {
  //      CHECK: vector.extractelement {{.*}}[3 : i32] : vector<4x8x16xf32>
  %1 = vector.extractelement %arg0[3 : i32] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extractelement {{.*}}[3 : i32, 3 : i32] : vector<4x8x16xf32>
  %2 = vector.extractelement %arg0[3 : i32, 3 : i32] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extractelement {{.*}}[3 : i32, 3 : i32, 3 : i32] : vector<4x8x16xf32>
  %3 = vector.extractelement %arg0[3 : i32, 3 : i32, 3 : i32] : vector<4x8x16xf32>
  return %1, %2, %3 : vector<8x16xf32>, vector<16xf32>, f32
}

// CHECK-LABEL: insertelement
func @insertelement(%a: f32, %b: vector<16xf32>, %c: vector<8x16xf32>, %res: vector<4x8x16xf32>) {
  //      CHECK: vector.insertelement %{{.*}}, %{{.*}}[3 : i32] : vector<8x16xf32> into vector<4x8x16xf32>
  %1 = vector.insertelement %c, %res[3 : i32] : vector<8x16xf32> into vector<4x8x16xf32>
  //      CHECK: vector.insertelement %{{.*}}, %{{.*}}[3 : i32, 3 : i32] : vector<16xf32> into vector<4x8x16xf32>
  %2 = vector.insertelement %b, %res[3 : i32, 3 : i32] : vector<16xf32> into vector<4x8x16xf32>
  //      CHECK: vector.insertelement %{{.*}}, %{{.*}}[3 : i32, 3 : i32, 3 : i32] : f32 into vector<4x8x16xf32>
  %3 = vector.insertelement %a, %res[3 : i32, 3 : i32, 3 : i32] : f32 into vector<4x8x16xf32>
  return
}

// CHECK-LABEL: outerproduct
func @outerproduct(%arg0: vector<4xf32>, %arg1: vector<8xf32>, %arg2: vector<4x8xf32>) -> vector<4x8xf32> {
  //     CHECK: vector.outerproduct {{.*}} : vector<4xf32>, vector<8xf32>
  %0 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<8xf32>
  //     CHECK: vector.outerproduct {{.*}}, {{.*}}, {{.*}} : vector<4xf32>, vector<8xf32>
  %1 = vector.outerproduct %arg0, %arg1, %arg2 : vector<4xf32>, vector<8xf32>
  return %1 : vector<4x8xf32>
}

// CHECK-LABEL: strided_slice
func @strided_slice(%arg0: vector<4x8x16xf32>) -> vector<2x2x16xf32> {
  //      CHECK: vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8x16xf32>
  %1 = vector.strided_slice %arg0 {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
  return %1: vector<2x2x16xf32>
}

// CHECK-LABEL: contraction
func @contraction(%arg0 : vector<7x8x16x15xf32>, %arg1 : vector<8x16x7x5xf32>,
                  %arg2 : vector<8x15x5xf32>, %arg3 : vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // Test contraction with batch and contracting dims.
  // CHECK: vector.contract {{.*}}, {{.*}}, {{.*}} {batch_dim_map = {{.*}}1, 0{{.*}}, contracting_dim_map = {{.*}}0, 2{{.*}}, {{.*}}2, 1{{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  %0 = vector.contract %arg0, %arg1, %arg2
    { batch_dim_map = [[1, 0]], contracting_dim_map = [[0, 2], [2, 1]] }
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>

  // Test contraction with only contracting dims. In this case the lhs/rhs
  // dimension of size 8 will be considered a free dim for lhs/rhs and will
  // appear twice in the output.
  // CHECK: vector.contract {{.*}}, {{.*}}, {{.*}} {contracting_dim_map = {{.*}}0, 2{{.*}}, {{.*}}2, 1{{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
  %1 = vector.contract %arg0, %arg1, %arg3
    { contracting_dim_map = [[0, 2], [2, 1]] }
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>

  // Test contraction with optional vector mask arguments.
  %lhs_mask = vector.make_index_tuple %arg4, %arg4, %arg4, %arg4
    : tuple<index, index, index, index>
  %rhs_mask = vector.make_index_tuple %arg4, %arg4, %arg4, %arg4
    : tuple<index, index, index, index>
  // CHECK: vector.contract {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}} {contracting_dim_map = {{.*}}0, 2{{.*}}, {{.*}}2, 1{{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
  %2 = vector.contract %arg0, %arg1, %arg3, %lhs_mask, %rhs_mask
    { contracting_dim_map = [[0, 2], [2, 1]] }
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
  return
}
