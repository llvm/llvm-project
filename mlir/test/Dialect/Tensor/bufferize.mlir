// RUN: mlir-opt %s -tensor-bufferize | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 20 + s0 + d1)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 140 + d1 * 20 + d2 * 5 + d3 + s0)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG: #[[$MAP4:.*]] = affine_map<() -> (1)>
// CHECK-DAG: #[[$MAP5:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>

// CHECK-LABEL:   func @dim(
// CHECK-SAME:              %[[TENSOR:.*]]: tensor<f32>,
// CHECK-SAME:              %[[INDEX:.*]]: index) -> index {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<f32>
// CHECK:           %[[EXTENT:.*]] = memref.dim %[[MEMREF]], %[[INDEX]] : memref<f32>
// CHECK:           return %[[EXTENT]] : index
func.func @dim(%arg0: tensor<f32>, %arg1: index) -> index {
  %0 = tensor.dim %arg0, %arg1 : tensor<f32>
  return %0 : index
}

// CHECK-LABEL: func @rank(
// CHECK-SAME:    %[[TENSOR:.*]]: tensor<*xf32>) -> index {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]]
// CHECK:           %[[EXTENT:.*]] = memref.rank %[[MEMREF]] : memref<*xf32>
func.func @rank(%arg0: tensor<*xf32>) -> index {
  %0 = tensor.rank %arg0 : tensor<*xf32>
  return %0 : index
}

// CHECK-LABEL:   func @tensor.cast(
// CHECK-SAME:                      %[[TENSOR:.*]]: tensor<?xindex>) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]]
// CHECK:           %[[CASTED:.*]] = memref.cast %[[MEMREF]] : memref<?xindex> to memref<2xindex>
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[CASTED]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func.func @tensor.cast(%arg0: tensor<?xindex>) -> tensor<2xindex> {
  %0 = tensor.cast %arg0 : tensor<?xindex> to tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL:   func @tensor.cast_from_unranked(
// CHECK-SAME:                                    %[[TENSOR:.*]]: tensor<*xf32>) -> tensor<2xf32> {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<*xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref.cast %[[MEMREF]] : memref<*xf32> to memref<2xf32>
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[CASTED_MEMREF]] : memref<2xf32>
// CHECK:           return %[[RET]] : tensor<2xf32>
func.func @tensor.cast_from_unranked(%arg0: tensor<*xf32>) -> tensor<2xf32> {
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL:   func @tensor.cast_to_unranked(
// CHECK-SAME:                                  %[[TENSOR:.*]]: tensor<2xf32>) -> tensor<*xf32> {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<2xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref.cast %[[MEMREF]] : memref<2xf32> to memref<*xf32>
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[CASTED_MEMREF]] : memref<*xf32>
// CHECK:           return %[[RET]] : tensor<*xf32>
func.func @tensor.cast_to_unranked(%arg0: tensor<2xf32>) -> tensor<*xf32> {
  %0 = tensor.cast %arg0 : tensor<2xf32> to tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL:   func @tensor.extract(
// CHECK-SAME:                  %[[TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:                  %[[IDX:.*]]: index) -> f32 {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<?xf32>
// CHECK:           %[[RET:.*]] = memref.load %[[MEMREF]][%[[IDX]]] : memref<?xf32>
// CHECK:           return %[[RET]] : f32
// CHECK:         }
func.func @tensor.extract(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
  %0 = tensor.extract %arg0[%arg1] : tensor<?xf32>
  return %0 : f32
}

// CHECK-LABEL:   func @tensor.from_elements_no_elements() -> tensor<0xindex> {
// CHECK:           %[[RET:.*]] = arith.constant dense<> : tensor<0xindex>
// CHECK:           return %[[RET]] : tensor<0xindex>
func.func @tensor.from_elements_no_elements() -> tensor<0xindex> {
  %0 = tensor.from_elements : tensor<0xindex>
  return %0 : tensor<0xindex>
}

// CHECK-LABEL:   func @tensor.from_elements_0d(
// CHECK-SAME:        %[[ELEM0:.*]]: index) -> tensor<index> {
// CHECK:           %[[MEMREF:.*]] = memref.alloc() {{.*}} : memref<index>
// CHECK:           store %[[ELEM0]], %[[MEMREF]]
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<index>
func.func @tensor.from_elements_0d(%arg0: index) -> tensor<index> {
  %0 = tensor.from_elements %arg0 : tensor<index>
  return %0 : tensor<index>
}

// CHECK-LABEL:   func @tensor.from_elements_1d(
// CHECK-SAME:                               %[[ELEM0:.*]]: index,
// CHECK-SAME:                               %[[ELEM1:.*]]: index) -> tensor<2xindex> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[MEMREF:.*]] = memref.alloc() {{.*}} : memref<2xindex>
// CHECK:           store %[[ELEM0]], %[[MEMREF]][%[[C0]]]
// CHECK:           store %[[ELEM1]], %[[MEMREF]][%[[C1]]]
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func.func @tensor.from_elements_1d(%arg0: index, %arg1: index) -> tensor<2xindex> {
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL: func @tensor.from_elements_2d(
// CHECK-SAME:      %[[ELEM0:.*]]: index, %[[ELEM1:.*]]: index)
// CHECK-SAME:      -> tensor<3x2xindex> {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[MEMREF:.*]] = memref.alloc() {{.*}} : memref<3x2xindex>
// CHECK:         store %[[ELEM0]], %[[MEMREF]][%[[C0]], %[[C0]]]
// CHECK:         store %[[ELEM1]], %[[MEMREF]][%[[C0]], %[[C1]]]
// CHECK:         store %[[ELEM0]], %[[MEMREF]][%[[C1]], %[[C0]]]
// CHECK:         store %[[ELEM1]], %[[MEMREF]][%[[C1]], %[[C1]]]
// CHECK:         store %[[ELEM0]], %[[MEMREF]][%[[C2]], %[[C0]]]
// CHECK:         store %[[ELEM1]], %[[MEMREF]][%[[C2]], %[[C1]]]
// CHECK:         %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:         return %[[RET]] : tensor<3x2xindex>
func.func @tensor.from_elements_2d(%arg0: index, %arg1: index) -> tensor<3x2xindex> {
  %0 = tensor.from_elements %arg0, %arg1, %arg0, %arg1, %arg0, %arg1
         : tensor<3x2xindex>
  return %0 : tensor<3x2xindex>
}

// CHECK-LABEL: func @tensor.from_elements_3d(
//  CHECK-SAME:     %[[F0:.*]]: f32

// CHECK-DAG: %[[F1:.*]] = arith.constant 1.0{{0+}}e+00
// CHECK-DAG: %[[F2:.*]] = arith.constant 2.0
// CHECK-DAG: %[[F3:.*]] = arith.constant 3.0
// CHECK-DAG: %[[F4:.*]] = arith.constant 4.0
// CHECK-DAG: %[[F5:.*]] = arith.constant 5.0
// CHECK-DAG: %[[F6:.*]] = arith.constant 6.0
// CHECK-DAG: %[[F7:.*]] = arith.constant 7.0
// CHECK-DAG: %[[F8:.*]] = arith.constant 8.0
// CHECK-DAG: %[[F9:.*]] = arith.constant 9.0
// CHECK-DAG: %[[F10:.*]] = arith.constant 1.0{{0+}}e+01
// CHECK-DAG: %[[F11:.*]] = arith.constant 1.1{{0+}}e+01

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index

// CHECK-DAG: %[[MEMREF:.*]] = memref.alloc() {{.*}} : memref<3x2x2xf32>

// CHECK: store %[[F0]], %[[MEMREF]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK: store %[[F1]], %[[MEMREF]][%[[C0]], %[[C0]], %[[C1]]]
// CHECK: store %[[F2]], %[[MEMREF]][%[[C0]], %[[C1]], %[[C0]]]
// CHECK: store %[[F3]], %[[MEMREF]][%[[C0]], %[[C1]], %[[C1]]]
// CHECK: store %[[F4]], %[[MEMREF]][%[[C1]], %[[C0]], %[[C0]]]
// CHECK: store %[[F5]], %[[MEMREF]][%[[C1]], %[[C0]], %[[C1]]]
// CHECK: store %[[F6]], %[[MEMREF]][%[[C1]], %[[C1]], %[[C0]]]
// CHECK: store %[[F7]], %[[MEMREF]][%[[C1]], %[[C1]], %[[C1]]]
// CHECK: store %[[F8]], %[[MEMREF]][%[[C2]], %[[C0]], %[[C0]]]
// CHECK: store %[[F9]], %[[MEMREF]][%[[C2]], %[[C0]], %[[C1]]]
// CHECK: store %[[F10]], %[[MEMREF]][%[[C2]], %[[C1]], %[[C0]]]
// CHECK: store %[[F11]], %[[MEMREF]][%[[C2]], %[[C1]], %[[C1]]]

// CHECK: %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK: return %[[RET]] : tensor<3x2x2xf32>
func.func @tensor.from_elements_3d(%f0 : f32) -> tensor<3x2x2xf32> {
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %f4 = arith.constant 4.0 : f32
  %f5 = arith.constant 5.0 : f32
  %f6 = arith.constant 6.0 : f32
  %f7 = arith.constant 7.0 : f32
  %f8 = arith.constant 8.0 : f32
  %f9 = arith.constant 9.0 : f32
  %f10 = arith.constant 10.0 : f32
  %f11 = arith.constant 11.0 : f32
  %0 = tensor.from_elements %f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8,%f9,%f10,%f11
         : tensor<3x2x2xf32>
  return %0 : tensor<3x2x2xf32>
}

// CHECK-LABEL:   func @tensor.generate(
// CHECK-SAME:                                       %[[ARG:.*]]: tensor<*xf32>,
// CHECK-SAME:                                       %[[DYNAMIC_EXTENT:.*]]: index) -> tensor<?xindex> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[CASTED:.*]] = bufferization.to_memref %[[ARG]] : memref<*xf32>
// CHECK:           %[[MEMREF:.*]] = memref.alloc(%[[DYNAMIC_EXTENT]]) {{.*}} : memref<?xindex>
// CHECK:           scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[DYNAMIC_EXTENT]]) step (%[[C1]]) {
// CHECK:             %[[ELEM:.*]] = memref.dim %[[CASTED]], %[[I]] : memref<*xf32>
// CHECK:             store %[[ELEM]], %[[MEMREF]][%[[I]]] : memref<?xindex>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]] : memref<?xindex>
// CHECK:           return %[[RET]] : tensor<?xindex>
// CHECK:         }
func.func @tensor.generate(%arg: tensor<*xf32>, %dynamic_extent: index) -> tensor<?xindex> {
  %result = tensor.generate %dynamic_extent {
  ^bb0(%i : index):
    %elem = tensor.dim %arg, %i : tensor<*xf32>
    tensor.yield %elem : index
  } : tensor<?xindex>
  return %result : tensor<?xindex>
}

// Additional test that checks the logic for intermixed static and dynamic
// extents.
//
// CHECK-LABEL:   func @tensor.generate_static_and_dynamic(
// CHECK-SAME:        %[[DYNAMIC_EXTENT:.*]]: index) -> tensor<16x?xindex> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[MEMREF:.*]] = memref.alloc(%[[DYNAMIC_EXTENT]]) {{.*}} : memref<16x?xindex>
// CHECK:           scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[C16]], %[[DYNAMIC_EXTENT]]) step (%[[C1]], %[[C1]]) {
// CHECK:             %[[VAL_7:.*]] = arith.addi %[[I]], %[[J]] : index
// CHECK:             store %[[VAL_7]], %[[MEMREF]][%[[I]], %[[J]]] : memref<16x?xindex>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]] : memref<16x?xindex>
// CHECK:           return %[[RET]] : tensor<16x?xindex>
// CHECK:         }
func.func @tensor.generate_static_and_dynamic(%arg0: index) -> tensor<16x?xindex> {
  %result = tensor.generate %arg0 {
  ^bb0(%i: index, %j: index):
    %sum = arith.addi %i, %j : index
    tensor.yield %sum : index
  } : tensor<16x?xindex>
  return %result : tensor<16x?xindex>
}

// CHECK-LABEL: func @tensor.generate_unknown_ops_in_body
func.func @tensor.generate_unknown_ops_in_body(%arg0: index) -> tensor<?xindex> {
  // CHECK-NOT: tensor.generate
  %tensor = tensor.generate %arg0 {
  ^bb0(%iv: index):
    // CHECK: test.source
    %0 = "test.source"() : () -> index
    tensor.yield %0 : index
  } : tensor<?xindex>
  return %tensor : tensor<?xindex>
}

// CHECK-LABEL: func @tensor.extract_slice(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x?xf32>, %[[idx1:.*]]: index, %[[idx2:.*]]: index
func.func @tensor.extract_slice(
    %t1: tensor<?x?xf32>, %idx1: index, %idx2: index) -> tensor<?x10xf32> {
  // CHECK: %[[m:.*]] = bufferization.to_memref %[[t1]] : memref<?x?xf32>
  // CHECK: %[[r:.*]] = memref.subview %[[m]][5, %[[idx2]]] [%[[idx1]], 10] [1, 1] : memref<?x?xf32> to memref<?x10xf32, #[[$MAP0]]>
  %0 = tensor.extract_slice %t1[5, %idx2][%idx1, 10][1, 1]
      : tensor<?x?xf32> to tensor<?x10xf32>
  // CHECK: %[[r_tensor:.*]] = bufferization.to_tensor %[[r]]
  // CHECK: return %[[r_tensor]]
  return %0 : tensor<?x10xf32>
}

// CHECK-LABEL: func @tensor.extract_slice_rank_reducing(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x10x?xf32>, %[[idx1:.*]]: index,
//  CHECK-SAME:     %[[idx2:.*]]: index
func.func @tensor.extract_slice_rank_reducing(
    %t1: tensor<?x10x?xf32>, %idx1: index, %idx2: index) -> tensor<?x15xf32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x10x?xf32>
  // CHECK: %[[r:.*]] = memref.subview %[[m1]][5, %[[idx1]], 10] [%[[idx2]], 1, 15] [1, 1, 1] : memref<?x10x?xf32> to memref<?x15xf32, #[[$MAP0]]>
  %0 = tensor.extract_slice %t1[5, %idx1, 10][%idx2, 1, 15][1, 1, 1]
      : tensor<?x10x?xf32> to tensor<?x15xf32>
  // CHECK: %[[r_tensor:.*]] = bufferization.to_tensor %[[r]]
  // CHECK: return %[[r_tensor]]
  return %0 : tensor<?x15xf32>
}

// CHECK-LABEL: func @tensor.insert_slice(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x?xf32>, %[[t2:.*]]: tensor<?x10xf32>,
//  CHECK-SAME:     %[[idx1:.*]]: index, %[[idx2:.*]]: index
func.func @tensor.insert_slice(%t1: tensor<?x?xf32>, %t2: tensor<?x10xf32>,
                          %idx1: index, %idx2: index) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x?xf32>
  // CHECK-DAG: %[[m2:.*]] = bufferization.to_memref %[[t2]] : memref<?x10xf32>
  //     CHECK: %[[dim0:.*]] = memref.dim %[[m1]], %[[c0]]
  //     CHECK: %[[dim1:.*]] = memref.dim %[[m1]], %[[c1]]
  //     CHECK: %[[alloc:.*]] = memref.alloc(%[[dim0]], %[[dim1]])
  //     CHECK: memref.copy %[[m1]], %[[alloc]]
  //     CHECK: %[[subview:.*]] = memref.subview %[[alloc]][%[[idx1]], 5] [%[[idx2]], 10] [1, 1]
  //     CHECK: memref.copy %[[m2]], %[[subview]]
  %0 = tensor.insert_slice %t2 into %t1[%idx1, 5][%idx2, 10][1, 1]
      : tensor<?x10xf32> into tensor<?x?xf32>

  //     CHECK: %[[r:.*]] = bufferization.to_tensor %[[alloc]]
  //     CHECK: return %[[r]]
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @tensor.insert(
//  CHECK-SAME:     %[[t1:.*]]: tensor<5xf32>, %[[idx1:.*]]: index,
//  CHECK-SAME:     %[[f:.*]]: f32
func.func @tensor.insert(%t1: tensor<5xf32>, %idx1: index, %f: f32) -> tensor<5xf32> {
  // CHECK-DAG: %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32>
  // CHECK-DAG: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<5xf32>
  // CHECK: memref.copy %[[m1]], %[[alloc]]
  // CHECK: memref.store %[[f]], %[[alloc]][%[[idx1]]]
  %0 = tensor.insert %f into %t1[%idx1] : tensor<5xf32>

  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[alloc]]
  // CHECK: return %[[r]]
  return %0 : tensor<5xf32>
}

// CHECK-LABEL: func @tensor.expand_shape(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x10xf32>
func.func @tensor.expand_shape(%t1: tensor<?x10xf32>) -> tensor<2x?x10xf32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x10xf32>
  // CHECK: %[[expanded:.*]] = memref.expand_shape %[[m1]] [
  // CHECK-SAME: [0, 1], [2]] : memref<?x10xf32> into memref<2x?x10xf32>
  %0 = tensor.expand_shape %t1 [[0, 1], [2]]
      : tensor<?x10xf32> into tensor<2x?x10xf32>

  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[expanded]]
  // CHECK: return %[[r]]
  return %0 : tensor<2x?x10xf32>
}

// CHECK-LABEL: func @tensor.expand_shape_of_slice(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x20xf32>
func.func @tensor.expand_shape_of_slice(
    %t1: tensor<?x20xf32>, %o1: index, %s1: index) -> tensor<?x7x2x5xf32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x20xf32>
  // CHECK: %[[subview:.*]] = memref.subview %[[m1]][%{{.*}}, 5] [%{{.*}}, 10] [1, 1] : memref<?x20xf32> to memref<?x10xf32, #[[$MAP1]]>
  %0 = tensor.extract_slice %t1[%o1, 5][%s1, 10][1, 1] :
      tensor<?x20xf32> to tensor<?x10xf32>
  // CHECK: %[[expanded:.*]] = memref.expand_shape %[[subview]] [
  // CHECK-SAME: [0, 1], [2, 3]] : memref<?x10xf32, #[[$MAP1]]> into memref<?x7x2x5xf32, #[[$MAP2]]>
  %1 = tensor.expand_shape %0 [[0, 1], [2, 3]] :
      tensor<?x10xf32> into tensor<?x7x2x5xf32>
  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[expanded]]
  // CHECK: return %[[r]]
  return %1 : tensor<?x7x2x5xf32>
}

// CHECK-LABEL: func @tensor.expand_shape_of_slice2(
//  CHECK-SAME:     %[[t1:.*]]: tensor<1x2xf32>
func.func @tensor.expand_shape_of_slice2(%t1: tensor<1x2xf32>) -> tensor<1xf32> {
  // CHECK: memref.subview {{.*}} : memref<1x2xf32> to memref<1x1xf32, #[[$MAP5]]>
  %0 = tensor.extract_slice %t1[0, 0][1, 1][1, 1] : tensor<1x2xf32> to tensor<1x1xf32>
  // CHECK: memref.collapse_shape %{{.*}} [
  // CHECK-SAME: [0, 1]] : memref<1x1xf32, #[[$MAP5]]> into memref<1xf32>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
  return %1 : tensor<1xf32>
}

// CHECK-LABEL: func @tensor.collapse_shape(
//  CHECK-SAME:     %[[t1:.*]]: tensor<2x?x?xf32>
func.func @tensor.collapse_shape(%t1: tensor<2x?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<2x?x?xf32>
  // CHECK: %[[collapsed:.*]] = memref.collapse_shape %[[m1]] [
  // CHECK-SAME: [0, 1], [2]] : memref<2x?x?xf32> into memref<?x?xf32>
  %0 = tensor.collapse_shape %t1 [[0, 1], [2]]
      : tensor<2x?x?xf32> into tensor<?x?xf32>

  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[collapsed]]
  // CHECK: return %[[r]]
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @tensor.collapse_shape_to_scalar(
//  CHECK-SAME:     %[[t1:.*]]: tensor<1x1x1xf32>
func.func @tensor.collapse_shape_to_scalar(%t1: tensor<1x1x1xf32>) -> tensor<f32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<1x1x1xf32>
  // CHECK: %[[collapsed:.*]] = memref.collapse_shape %[[m1]] [] : memref<1x1x1xf32> into memref<f32>
  %0 = tensor.collapse_shape %t1 []
      : tensor<1x1x1xf32> into tensor<f32>

  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[collapsed]]
  // CHECK: return %[[r]]
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @tensor.collapse_shape_of_slice(
func.func @tensor.collapse_shape_of_slice(%arg0: tensor<2xi32>) -> tensor<i32> {
  // CHECK: memref.subview %{{.*}}[1] [1] [1] : memref<2xi32> to memref<1xi32, #[[$MAP3]]>
  %0 = tensor.extract_slice %arg0[1] [1] [1] : tensor<2xi32> to tensor<1xi32>
  // CHECK: memref.collapse_shape %{{.*}} [] : memref<1xi32, #[[$MAP3]]> into memref<i32, #[[$MAP4]]>
  %1 = tensor.collapse_shape %0 [] : tensor<1xi32> into tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @tensor.collapse_shape_of_slice2(
func.func @tensor.collapse_shape_of_slice2(
    %arg0: tensor<?x?x?x?xi64>, %o1: index, %o2: index, %o3: index, %o4: index)
    -> tensor<87x63648xi64> {
  // CHECK: %[[subview:.*]] = memref.subview %{{.*}} : memref<?x?x?x?xi64> to memref<87x78x68x12xi64, #{{.*}}>
  %0 = tensor.extract_slice %arg0[%o1, %o2, %o3, %o4] [87, 78, 68, 12] [1, 1, 1, 1] : tensor<?x?x?x?xi64> to tensor<87x78x68x12xi64>

  // This memref may not be collapsible, so the buffer must be copied to get rid
  // of the layout map.
  // CHECK: %[[alloc:.*]] = memref.alloc() {{.*}} : memref<87x78x68x12xi64>
  // CHECK: memref.copy %[[subview]], %[[alloc]]
  // CHECK: memref.collapse_shape %[[alloc]] [
  // CHECK-SAME: [0], [1, 2, 3]] : memref<87x78x68x12xi64> into memref<87x63648xi64>
  %1 = tensor.collapse_shape %0 [[0], [1, 2, 3]] : tensor<87x78x68x12xi64> into tensor<87x63648xi64>
  return %1 : tensor<87x63648xi64>
}
