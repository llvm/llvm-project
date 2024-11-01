// RUN: mlir-opt %s -tensor-bufferize -cse -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @dim(
// CHECK-SAME:              %[[TENSOR:.*]]: tensor<*xf32>,
// CHECK-SAME:              %[[INDEX:.*]]: index) -> index {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<*xf32>
// CHECK:           %[[EXTENT:.*]] = memref.dim %[[MEMREF]], %[[INDEX]] : memref<*xf32>
// CHECK:           return %[[EXTENT]] : index
func.func @dim(%arg0: tensor<*xf32>, %arg1: index) -> index {
  %0 = tensor.dim %arg0, %arg1 : tensor<*xf32>
  return %0 : index
}

// -----

// CHECK-LABEL: func @rank(
// CHECK-SAME:    %[[TENSOR:.*]]: tensor<*xf32>) -> index {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]]
// CHECK:           %[[EXTENT:.*]] = memref.rank %[[MEMREF]] : memref<*xf32>
func.func @rank(%arg0: tensor<*xf32>) -> index {
  %0 = tensor.rank %arg0 : tensor<*xf32>
  return %0 : index
}

// -----

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

// -----

// CHECK-LABEL:   func @tensor.cast_from_unranked(
// CHECK-SAME:                                    %[[TENSOR:.*]]: tensor<*xf32>) -> tensor<2xf32> {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<*xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref.cast %[[MEMREF]] : memref<*xf32> to memref<2xf32, strided<[?], offset: ?>>
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[CASTED_MEMREF]] : memref<2xf32, strided<[?], offset: ?>>
// CHECK:           return %[[RET]] : tensor<2xf32>
func.func @tensor.cast_from_unranked(%arg0: tensor<*xf32>) -> tensor<2xf32> {
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

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

// -----
func.func @tensor.empty() -> tensor<5xf32> {
  // expected-error@+2 {{failed to bufferize op}}
  // expected-error@+1 {{cannot be bufferized, but can be converted to bufferization.alloc_tensor}}
  %0 = tensor.empty() : tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

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

// -----

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

// -----

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

// -----

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

// -----

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

// -----

// CHECK-LABEL:   func @tensor.generate(
// CHECK-SAME:        %[[ARG:.*]]: tensor<*xf32>,
// CHECK-SAME:        %[[DYNAMIC_EXTENT:.*]]: index) -> tensor<?xindex> {
// CHECK-DAG:       %[[ARG_M:.*]] = bufferization.to_memref %[[ARG]] : memref<*xf32>
// CHECK-DAG:       %[[ALLOC:.*]] = memref.alloc(%[[DYNAMIC_EXTENT]]) {{.*}} : memref<?xindex>
// CHECK:           %[[ALLOC_T:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK:           %[[MAPPED:.*]] = linalg.map
// CHECK:                 outs(%[[ALLOC_T]] : tensor<?xindex>)
// CHECK:             %[[INDEX:.*]] = linalg.index 0 : index
// CHECK:             %[[ELEM:.*]] = memref.dim %[[ARG_M]], %[[INDEX]] : memref<*xf32>
// CHECK:             linalg.yield %[[ELEM]]
// CHECK:           }
// CHECK:           return %[[MAPPED]] : tensor<?xindex>
// CHECK:         }
func.func @tensor.generate(%arg: tensor<*xf32>, %dynamic_extent: index) -> tensor<?xindex> {
  %result = tensor.generate %dynamic_extent {
  ^bb0(%i : index):
    %elem = tensor.dim %arg, %i : tensor<*xf32>
    tensor.yield %elem : index
  } : tensor<?xindex>
  return %result : tensor<?xindex>
}

// -----

// Additional test that checks the logic for intermixed static and dynamic
// extents.
//
// CHECK-LABEL:   func @tensor.generate_static_and_dynamic(
// CHECK-SAME:        %[[DYNAMIC_EXTENT:.*]]: index) -> tensor<16x?xindex> {
// CHECK:           %[[ALLOC:.*]] = memref.alloc(%[[DYNAMIC_EXTENT]]) {{.*}} : memref<16x?xindex>
// CHECK:           %[[ALLOC_T:.*]] = bufferization.to_tensor %[[ALLOC]]
// CHECK:           %[[MAPPED:.*]] = linalg.map
// CHECK:                 outs(%[[ALLOC_T]] : tensor<16x?xindex>)
// CHECK:             %[[INDEX0:.*]] = linalg.index 0
// CHECK:             %[[INDEX1:.*]] = linalg.index 1
// CHECK:             %[[ADD:.*]] = arith.addi %[[INDEX0]], %[[INDEX1]]
// CHECK:             linalg.yield %[[ADD]]
// CHECK:           }
// CHECK:           return %[[MAPPED]] : tensor<16x?xindex>
// CHECK:         }
func.func @tensor.generate_static_and_dynamic(%arg0: index) -> tensor<16x?xindex> {
  %result = tensor.generate %arg0 {
  ^bb0(%i: index, %j: index):
    %sum = arith.addi %i, %j : index
    tensor.yield %sum : index
  } : tensor<16x?xindex>
  return %result : tensor<16x?xindex>
}

// -----

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

// -----

// CHECK-LABEL: func @tensor.extract_slice(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x?xf32>, %[[idx1:.*]]: index, %[[idx2:.*]]: index
func.func @tensor.extract_slice(
    %t1: tensor<?x?xf32>, %idx1: index, %idx2: index) -> tensor<?x10xf32> {
  // CHECK: %[[m:.*]] = bufferization.to_memref %[[t1]] : memref<?x?xf32>
  // CHECK: %[[r:.*]] = memref.subview %[[m]][5, %[[idx2]]] [%[[idx1]], 10] [1, 1] : memref<?x?xf32> to memref<?x10xf32, strided<[?, 1], offset: ?>>
  %0 = tensor.extract_slice %t1[5, %idx2][%idx1, 10][1, 1]
      : tensor<?x?xf32> to tensor<?x10xf32>
  // CHECK: %[[r_tensor:.*]] = bufferization.to_tensor %[[r]]
  // CHECK: return %[[r_tensor]]
  return %0 : tensor<?x10xf32>
}

// -----

// CHECK-LABEL: func @tensor.extract_slice_rank_reducing(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x10x?xf32>, %[[idx1:.*]]: index,
//  CHECK-SAME:     %[[idx2:.*]]: index
func.func @tensor.extract_slice_rank_reducing(
    %t1: tensor<?x10x?xf32>, %idx1: index, %idx2: index) -> tensor<?x15xf32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x10x?xf32>
  // CHECK: %[[r:.*]] = memref.subview %[[m1]][5, %[[idx1]], 10] [%[[idx2]], 1, 15] [1, 1, 1] : memref<?x10x?xf32> to memref<?x15xf32, strided<[?, 1], offset: ?>>
  %0 = tensor.extract_slice %t1[5, %idx1, 10][%idx2, 1, 15][1, 1, 1]
      : tensor<?x10x?xf32> to tensor<?x15xf32>
  // CHECK: %[[r_tensor:.*]] = bufferization.to_tensor %[[r]]
  // CHECK: return %[[r_tensor]]
  return %0 : tensor<?x15xf32>
}

// -----

// CHECK-LABEL: func @tensor.insert_slice(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x?xf32>, %[[t2:.*]]: tensor<?x10xf32>,
//  CHECK-SAME:     %[[idx1:.*]]: index, %[[idx2:.*]]: index
func.func @tensor.insert_slice(%t1: tensor<?x?xf32>, %t2: tensor<?x10xf32>,
                               %idx1: index, %idx2: index) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x?xf32>
  // CHECK-DAG: %[[m2:.*]] = bufferization.to_memref %[[t2]] : memref<?x10xf32>
  // CHECK-DAG: %[[dim0:.*]] = memref.dim %[[m1]], %[[c0]]
  // CHECK-DAG: %[[dim1:.*]] = memref.dim %[[m1]], %[[c1]]
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

// -----

// CHECK-LABEL: func @tensor.insert_slice_rank_reducing_1(
func.func @tensor.insert_slice_rank_reducing_1(
    %t1: tensor<?x?xf32>, %f: tensor<f32>, %idx1: index, %idx2: index)
  -> tensor<?x?xf32>
{
  // CHECK: %[[alloc:.*]] = memref.alloc{{.*}} : memref<?x?xf32>
  // CHECK: memref.subview %[[alloc]][%{{.*}}, %{{.*}}] [1, 1] [1, 1] : memref<?x?xf32> to memref<f32, strided<[], offset: ?>>
  // CHECK: memref.copy {{.*}} : memref<f32> to memref<f32, strided<[], offset: ?>>
  %0 = tensor.insert_slice %f into %t1[%idx1, %idx2][1, 1][1, 1]
      : tensor<f32> into tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @tensor.insert_slice_rank_reducing_2(
func.func @tensor.insert_slice_rank_reducing_2(
    %t1: tensor<?x?x?x?x?x?x?xf32>, %t2: tensor<2x1x4x1x1xf32>, %i: index)
  -> tensor<?x?x?x?x?x?x?xf32>
{
  // CHECK: %[[alloc:.*]] = memref.alloc{{.*}} : memref<?x?x?x?x?x?x?xf32>
  // CHECK: memref.subview %[[alloc]][{{.*}}] [1, 2, 1, 4, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1] : memref<?x?x?x?x?x?x?xf32> to memref<2x1x4x1x1xf32, strided<[?, ?, ?, ?, ?], offset: ?>>
  // CHECK: memref.copy {{.*}} : memref<2x1x4x1x1xf32> to memref<2x1x4x1x1xf32, strided<[?, ?, ?, ?, ?], offset: ?>>
  %0 = tensor.insert_slice %t2 into %t1[%i, %i, %i, %i, %i, %i, %i][1, 2, 1, 4, 1, 1, 1][1, 1, 1, 1, 1, 1, 1]
      : tensor<2x1x4x1x1xf32> into tensor<?x?x?x?x?x?x?xf32>
  return %0 : tensor<?x?x?x?x?x?x?xf32>
}

// -----

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

// -----

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

// -----

// CHECK-LABEL: func @tensor.expand_shape_of_slice(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x20xf32>
func.func @tensor.expand_shape_of_slice(
    %t1: tensor<?x20xf32>, %o1: index, %s1: index) -> tensor<?x7x2x5xf32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x20xf32>
  // CHECK: %[[subview:.*]] = memref.subview %[[m1]][%{{.*}}, 5] [%{{.*}}, 10] [1, 1] : memref<?x20xf32> to memref<?x10xf32, strided<[20, 1], offset: ?>>
  %0 = tensor.extract_slice %t1[%o1, 5][%s1, 10][1, 1] :
      tensor<?x20xf32> to tensor<?x10xf32>
  // CHECK: %[[expanded:.*]] = memref.expand_shape %[[subview]] [
  // CHECK-SAME: [0, 1], [2, 3]] : memref<?x10xf32, strided<[20, 1], offset: ?>> into memref<?x7x2x5xf32, strided<[140, 20, 5, 1], offset: ?>>
  %1 = tensor.expand_shape %0 [[0, 1], [2, 3]] :
      tensor<?x10xf32> into tensor<?x7x2x5xf32>
  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[expanded]]
  // CHECK: return %[[r]]
  return %1 : tensor<?x7x2x5xf32>
}

// -----

// CHECK-LABEL: func @tensor.expand_shape_of_scalar_slice(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?xf32>
func.func @tensor.expand_shape_of_scalar_slice(
    %t1: tensor<?xf32>, %o1: index, %s1: index) -> tensor<1xf32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?xf32>
  // CHECK: %[[subview:.*]] = memref.subview %[[m1]][%{{.*}}] [1] [1] :  memref<?xf32> to memref<f32, strided<[], offset: ?>>
  %0 = tensor.extract_slice %t1[%o1][1][1] : tensor<?xf32> to tensor<f32>
  // CHECK: %[[expanded:.*]] = memref.expand_shape %[[subview]] [] : memref<f32, strided{{.*}}> into memref<1xf32, strided<[1], offset: ?>>
  %1 = tensor.expand_shape %0 [] : tensor<f32> into tensor<1xf32>
  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[expanded]]
  // CHECK: return %[[r]]
  return %1 : tensor<1xf32>
}

// -----

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

// -----

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

// -----

// CHECK-LABEL: func @tensor.collapse_shape_of_slice(
func.func @tensor.collapse_shape_of_slice(%arg0: tensor<2xi32>) -> tensor<i32> {
  // CHECK: memref.subview %{{.*}}[1] [1] [1] : memref<2xi32> to memref<1xi32, strided<[1], offset: 1>>
  %0 = tensor.extract_slice %arg0[1] [1] [1] : tensor<2xi32> to tensor<1xi32>
  // CHECK: memref.collapse_shape %{{.*}} [] : memref<1xi32, strided<[1], offset: 1>> into memref<i32, strided<[], offset: 1>>
  %1 = tensor.collapse_shape %0 [] : tensor<1xi32> into tensor<i32>
  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: func @tensor.collapse_shape_of_slice2(
func.func @tensor.collapse_shape_of_slice2(
    %arg0: tensor<?x?x?x?xi64>, %o1: index, %o2: index, %o3: index, %o4: index)
    -> tensor<87x63648xi64> {
  // CHECK: %[[subview:.*]] = memref.subview %{{.*}} : memref<?x?x?x?xi64> to memref<87x78x68x12xi64, strided{{.*}}>
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

// -----

// CHECK-LABEL: func @tensor.collapse_shape_of_slice3(
//  CHECK-SAME:     %[[t1:.*]]: tensor<1x2xf32>
func.func @tensor.collapse_shape_of_slice3(%t1: tensor<1x2xf32>) -> tensor<1xf32> {
  // CHECK: memref.subview {{.*}} : memref<1x2xf32> to memref<1x1xf32, strided<[2, 1]>>
  %0 = tensor.extract_slice %t1[0, 0][1, 1][1, 1] : tensor<1x2xf32> to tensor<1x1xf32>
  // CHECK: memref.collapse_shape %{{.*}} [
  // CHECK-SAME: [0, 1]] : memref<1x1xf32, strided<[2, 1]>> into memref<1xf32, strided<[2]>>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<1x1xf32> into tensor<1xf32>
  return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL:   func @tensor.collapse_shape_of_slice4(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x2x4xf32>,
// CHECK-SAME:      %[[OFFSET:.*]]: index) -> tensor<8xf32> {
func.func @tensor.collapse_shape_of_slice4(%arg0: tensor<?x2x4xf32>, %offset: index, %size: index) -> tensor<8xf32> {
  // CHECK: memref.subview %{{.*}} : memref<?x2x4xf32> to memref<4x2x1xf32, strided<[8, 4, 1], offset: ?>>
  %0 = tensor.extract_slice %arg0[0, 0, %offset] [4, 2, 1] [1, 1, 1] : tensor<?x2x4xf32> to tensor<4x2x1xf32>
  // CHECK: memref.collapse_shape %{{.*}} [
  // CHECK-SAME: [0, 1, 2]] : memref<4x2x1xf32, strided<[8, 4, 1], offset: ?>> into memref<8xf32, strided<[4], offset: ?>>
  %ret = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<4x2x1xf32> into tensor<8xf32>
  return %ret: tensor<8xf32>
}

// -----

// CHECK-LABEL: func @tensor.collapse_shape_of_slice5(
func.func @tensor.collapse_shape_of_slice5(%arg0: tensor<2x2x2xi64>) -> tensor<4xi64> {
  // CHECK: %[[subview:.*]] = memref.subview %{{.*}} : memref<2x2x2xi64> to memref<2x1x2xi64, {{.*}}>
  %0 = tensor.extract_slice %arg0[0, 0, 0] [2, 1, 2] [1, 1, 1] : tensor<2x2x2xi64> to tensor<2x1x2xi64>

  // This memref is not collapsible, so the buffer must be copied to get rid of
  // the layout map.
  // CHECK: %[[alloc:.*]] = memref.alloc() {{.*}} : memref<2x1x2xi64>
  // CHECK: memref.copy %[[subview]], %[[alloc]]
  // CHECK: memref.collapse_shape %[[alloc]] [
  // CHECK-SAME: [0, 1, 2]] : memref<2x1x2xi64> into memref<4xi64>
  %1 = tensor.collapse_shape %0 [[0, 1, 2]] : tensor<2x1x2xi64> into tensor<4xi64>
  return %1 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @tensor.reshape(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x10xf32>
func.func @tensor.reshape(%t1: tensor<?x10xf32>) -> tensor<2x2x5xf32> {
  // CHECK: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x10xf32>

  // CHECK: %[[two:.*]] = arith.constant 2 : i64
  %two = arith.constant 2 : i64
  // CHECK: %[[five:.*]] = arith.constant 5 : i64
  %five = arith.constant 5 : i64

  // CHECK: %[[alloc:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3xi64>
  // CHECK: %[[zero_idx:.*]] = arith.constant 0 : index
  // CHECK: %[[one_idx:.*]] = arith.constant 1 : index
  // CHECK: %[[two_idx:.*]] = arith.constant 2 : index
  // CHECK: memref.store %[[two]], %[[alloc]][%[[zero_idx]]] : memref<3xi64>
  // CHECK: memref.store %[[two]], %[[alloc]][%[[one_idx]]] : memref<3xi64>
  // CHECK: memref.store %[[five]], %[[alloc]][%[[two_idx]]] : memref<3xi64>
  %shape = tensor.from_elements %two, %two, %five : tensor<3xi64>

  // CHECK: %[[reshaped:.*]] = memref.reshape %[[m1]](%[[alloc]]) : (memref<?x10xf32>, memref<3xi64>) -> memref<2x2x5xf32>
  %reshaped = tensor.reshape %t1(%shape) : (tensor<?x10xf32>, tensor<3xi64>) -> tensor<2x2x5xf32>

  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[reshaped]]
  // CHECK: return %[[r]]
  return %reshaped : tensor<2x2x5xf32>
}

// -----

// CHECK:       #[[$sum_map_1:.+]] = affine_map<()[s0, s1] -> (s1 + s0 + 5)>
// CHECK:       #[[$sum_map_2:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 10)>
// CHECK-LABEL: func @tensor.pad(
//  CHECK-SAME:   %[[t1:.*]]: tensor<?x10xindex>, %[[l2:.*]]: index, %[[h1:.*]]: index, %[[h2:.*]]: index
func.func @tensor.pad(%t1: tensor<?x10xindex>, %l2: index, %h1: index,
                      %h2: index) -> tensor<?x?xindex> {
  // CHECK-DAG: %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<?x10xindex>
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[dim0:.*]] = memref.dim %[[m1]], %[[c0]]
  // CHECK-DAG: %[[dim1:.*]] = memref.dim %[[m1]], %[[c1]]
  // CHECK-DAG: %[[size0:.*]] = affine.apply #[[$sum_map_1]]()[%[[h1]], %[[dim0]]]
  // CHECK-DAG: %[[size1:.*]] = affine.apply #[[$sum_map_2]]()[%[[l2]], %[[h2]]]
  // CHECK:     %[[alloc:.*]] = memref.alloc(%[[size0]], %[[size1]]) {{.*}} : memref<?x?xindex>
  // CHECK:     %[[alloc_t:.*]] = bufferization.to_tensor %[[alloc]]
  // CHECK:     %[[mapped:.*]] = linalg.map
  // CHECK:           outs(%[[alloc_t]] : tensor<?x?xindex>)
  // CHECK:       %[[index0:.*]] = linalg.index 0
  // CHECK:       %[[index1:.*]] = linalg.index 1
  // CHECK:       %[[mul:.*]] = arith.muli %[[index0]], %[[index1]]
  // CHECK:       linalg.yield %[[mul]]
  // CHECK:     }
  // CHECK:     %[[mapped_m:.*]] = bufferization.to_memref %[[mapped]]
  // CHECK:     %[[subview:.*]] = memref.subview %[[mapped_m]][5, %[[l2]]] [%[[dim0]], 10] [1, 1]
  // CHECK:     memref.copy %[[m1]], %[[subview]]
  %0 = tensor.pad %t1 low[5, %l2] high[%h1, %h2] {
  ^bb0(%arg0: index, %arg1: index):
    %m = arith.muli %arg0, %arg1 : index
    tensor.yield %m : index
  } : tensor<?x10xindex> to tensor<?x?xindex>

  // CHECK:     %[[r:.*]] = bufferization.to_tensor %[[mapped_m]]
  // CHECK:     return %[[r]] : tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}
