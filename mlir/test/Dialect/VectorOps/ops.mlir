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

// CHECK-LABEL: @vector_broadcast
func @vector_broadcast(%a: f32, %b: vector<16xf32>, %c: vector<1x16xf32>, %d: vector<8x1xf32>) -> vector<8x16xf32> {
  //      CHECK: vector.broadcast %{{.*}} : f32 to vector<16xf32>
  %0 = vector.broadcast %a : f32 to vector<16xf32>
  //      CHECK-NEXT: vector.broadcast %{{.*}} : vector<16xf32> to vector<8x16xf32>
  %1 = vector.broadcast %b : vector<16xf32> to vector<8x16xf32>
  //      CHECK-NEXT: vector.broadcast %{{.*}} : vector<1x16xf32> to vector<8x16xf32>
  %2 = vector.broadcast %c : vector<1x16xf32> to vector<8x16xf32>
  //      CHECK-NEXT: vector.broadcast %{{.*}} : vector<8x1xf32> to vector<8x16xf32>
  %3 = vector.broadcast %d : vector<8x1xf32> to vector<8x16xf32>
  return %3 : vector<8x16xf32>
}

// CHECK-LABEL: @extractelement
func @extractelement(%arg0: vector<4x8x16xf32>) -> (vector<8x16xf32>, vector<16xf32>, f32) {
  //      CHECK: vector.extractelement {{.*}}[3 : i32] : vector<4x8x16xf32>
  %1 = vector.extractelement %arg0[3 : i32] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extractelement {{.*}}[3 : i32, 3 : i32] : vector<4x8x16xf32>
  %2 = vector.extractelement %arg0[3 : i32, 3 : i32] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extractelement {{.*}}[3 : i32, 3 : i32, 3 : i32] : vector<4x8x16xf32>
  %3 = vector.extractelement %arg0[3 : i32, 3 : i32, 3 : i32] : vector<4x8x16xf32>
  return %1, %2, %3 : vector<8x16xf32>, vector<16xf32>, f32
}

// CHECK-LABEL: @insertelement
func @insertelement(%a: f32, %b: vector<16xf32>, %c: vector<8x16xf32>, %res: vector<4x8x16xf32>) {
  //      CHECK: vector.insertelement %{{.*}}, %{{.*}}[3 : i32] : vector<8x16xf32> into vector<4x8x16xf32>
  %1 = vector.insertelement %c, %res[3 : i32] : vector<8x16xf32> into vector<4x8x16xf32>
  //      CHECK: vector.insertelement %{{.*}}, %{{.*}}[3 : i32, 3 : i32] : vector<16xf32> into vector<4x8x16xf32>
  %2 = vector.insertelement %b, %res[3 : i32, 3 : i32] : vector<16xf32> into vector<4x8x16xf32>
  //      CHECK: vector.insertelement %{{.*}}, %{{.*}}[3 : i32, 3 : i32, 3 : i32] : f32 into vector<4x8x16xf32>
  %3 = vector.insertelement %a, %res[3 : i32, 3 : i32, 3 : i32] : f32 into vector<4x8x16xf32>
  return
}

// CHECK-LABEL: @outerproduct
func @outerproduct(%arg0: vector<4xf32>, %arg1: vector<8xf32>, %arg2: vector<4x8xf32>) -> vector<4x8xf32> {
  //     CHECK: vector.outerproduct {{.*}} : vector<4xf32>, vector<8xf32>
  %0 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<8xf32>
  //     CHECK: vector.outerproduct {{.*}}, {{.*}}, {{.*}} : vector<4xf32>, vector<8xf32>
  %1 = vector.outerproduct %arg0, %arg1, %arg2 : vector<4xf32>, vector<8xf32>
  return %1 : vector<4x8xf32>
}

// CHECK-LABEL: @insert_strided_slice
func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  //      CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [2, 2, 2], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
  %1 = vector.insert_strided_slice %a, %b {offsets = [2, 2, 2], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
  return
}

// CHECK-LABEL: @strided_slice
func @strided_slice(%arg0: vector<4x8x16xf32>) -> vector<2x2x16xf32> {
  //      CHECK: vector.strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8x16xf32>
  %1 = vector.strided_slice %arg0 {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
  return %1: vector<2x2x16xf32>
}

#contraction_accesses0 = [
  (b0, f0, f1, c0, c1) -> (c0, b0, c1, f0),
  (b0, f0, f1, c0, c1) -> (b0, c1, c0, f1),
  (b0, f0, f1, c0, c1) -> (b0, f0, f1)
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
#contraction_accesses1 = [
  (f0, f1, f2, f3, c0, c1) -> (c0, f0, c1, f2),
  (f0, f1, f2, f3, c0, c1) -> (f1, c1, c0, f3),
  (f0, f1, f2, f3, c0, c1) -> (f0, f1, f2, f3)
]
#contraction_trait1 = {
  indexing_maps = #contraction_accesses1,
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction",
                    "reduction"]
}
// CHECK-LABEL: contraction
func @contraction(%arg0 : vector<7x8x16x15xf32>, %arg1 : vector<8x16x7x5xf32>,
                  %arg2 : vector<8x15x5xf32>, %arg3 : vector<8x15x8x5xf32>,
                  %arg4 : index) {
  // Test contraction with batch and contracting dims.
  // CHECK: vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} {{.*}}, {{.*}}, {{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  %0 = vector.contract #contraction_trait0 %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  // Test contraction with only contracting dims. In this case the lhs/rhs
  // dimension of size 8 will be considered a parallel dim for lhs/rhs and will
  // appear twice in the output.
  // CHECK: vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} {{.*}}, {{.*}}, {{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
  %1 = vector.contract #contraction_trait1 %arg0, %arg1, %arg3
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
  // Test contraction with optional vector mask arguments.
  %lhs_mask = vector.make_index_tuple %arg4, %arg4, %arg4, %arg4
    : tuple<index, index, index, index>
  %rhs_mask = vector.make_index_tuple %arg4, %arg4, %arg4, %arg4
    : tuple<index, index, index, index>
  // CHECK: vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
  %2 = vector.contract #contraction_trait1 %arg0, %arg1, %arg3, %lhs_mask,
                                           %rhs_mask
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x8x5xf32>
  return
}

// CHECK-LABEL: create_vector_mask
func @create_vector_mask() {
  // CHECK:      %[[C2:.*]] = constant 2 : index
  %c2 = constant 2 : index
  // CHECK-NEXT: %[[C3:.*]] = constant 3 : index
  %c3 = constant 3 : index
  // CHECK-NEXT: vector.create_mask %[[C3]], %[[C2]] : vector<4x3xi1>
  %0 = vector.create_mask %c3, %c2 : vector<4x3xi1>

  return
}
