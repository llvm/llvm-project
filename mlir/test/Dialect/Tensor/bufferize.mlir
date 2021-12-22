// RUN: mlir-opt %s -tensor-bufferize | FileCheck %s

// CHECK-LABEL:   func @dim(
// CHECK-SAME:              %[[TENSOR:.*]]: tensor<f32>,
// CHECK-SAME:              %[[INDEX:.*]]: index) -> index {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<f32>
// CHECK:           %[[EXTENT:.*]] = memref.dim %[[MEMREF]], %[[INDEX]] : memref<f32>
// CHECK:           return %[[EXTENT]] : index
func @dim(%arg0: tensor<f32>, %arg1: index) -> index {
  %0 = tensor.dim %arg0, %arg1 : tensor<f32>
  return %0 : index
}

// CHECK-LABEL: func @rank(
// CHECK-SAME:    %[[TENSOR:.*]]: tensor<*xf32>) -> index {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]]
// CHECK:           %[[EXTENT:.*]] = memref.rank %[[MEMREF]] : memref<*xf32>
func @rank(%arg0: tensor<*xf32>) -> index {
  %0 = tensor.rank %arg0 : tensor<*xf32>
  return %0 : index
}

// CHECK-LABEL:   func @tensor.cast(
// CHECK-SAME:                      %[[TENSOR:.*]]: tensor<?xindex>) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]]
// CHECK:           %[[CASTED:.*]] = memref.cast %[[MEMREF]] : memref<?xindex> to memref<2xindex>
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[CASTED]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor.cast(%arg0: tensor<?xindex>) -> tensor<2xindex> {
  %0 = tensor.cast %arg0 : tensor<?xindex> to tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL:   func @tensor.cast_from_unranked(
// CHECK-SAME:                                    %[[TENSOR:.*]]: tensor<*xf32>) -> tensor<2xf32> {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<*xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref.cast %[[MEMREF]] : memref<*xf32> to memref<2xf32>
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[CASTED_MEMREF]] : memref<2xf32>
// CHECK:           return %[[RET]] : tensor<2xf32>
func @tensor.cast_from_unranked(%arg0: tensor<*xf32>) -> tensor<2xf32> {
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL:   func @tensor.cast_to_unranked(
// CHECK-SAME:                                  %[[TENSOR:.*]]: tensor<2xf32>) -> tensor<*xf32> {
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<2xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref.cast %[[MEMREF]] : memref<2xf32> to memref<*xf32>
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[CASTED_MEMREF]] : memref<*xf32>
// CHECK:           return %[[RET]] : tensor<*xf32>
func @tensor.cast_to_unranked(%arg0: tensor<2xf32>) -> tensor<*xf32> {
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
func @tensor.extract(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
  %0 = tensor.extract %arg0[%arg1] : tensor<?xf32>
  return %0 : f32
}

// CHECK-LABEL:   func @tensor.from_elements_no_elements() -> tensor<0xindex> {
// CHECK:           %[[MEMREF:.*]] = memref.alloc() : memref<0xindex>
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<0xindex>
func @tensor.from_elements_no_elements() -> tensor<0xindex> {
  %0 = tensor.from_elements : tensor<0xindex>
  return %0 : tensor<0xindex>
}

// CHECK-LABEL:   func @tensor.from_elements_0d(
// CHECK-SAME:        %[[ELEM0:.*]]: index) -> tensor<index> {
// CHECK:           %[[MEMREF:.*]] = memref.alloc() : memref<index>
// CHECK:           store %[[ELEM0]], %[[MEMREF]]
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<index>
func @tensor.from_elements_0d(%arg0: index) -> tensor<index> {
  %0 = tensor.from_elements %arg0 : tensor<index>
  return %0 : tensor<index>
}

// CHECK-LABEL:   func @tensor.from_elements_1d(
// CHECK-SAME:                               %[[ELEM0:.*]]: index,
// CHECK-SAME:                               %[[ELEM1:.*]]: index) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = memref.alloc() : memref<2xindex>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           store %[[ELEM0]], %[[MEMREF]][%[[C0]]]
// CHECK:           store %[[ELEM1]], %[[MEMREF]][%[[C1]]]
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor.from_elements_1d(%arg0: index, %arg1: index) -> tensor<2xindex> {
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL: func @tensor.from_elements_2d(
// CHECK-SAME:      %[[ELEM0:.*]]: index, %[[ELEM1:.*]]: index)
// CHECK-SAME:      -> tensor<3x2xindex> {
// CHECK:         %[[MEMREF:.*]] = memref.alloc() : memref<3x2xindex>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         store %[[ELEM0]], %[[MEMREF]][%[[C0]], %[[C0]]]
// CHECK:         store %[[ELEM1]], %[[MEMREF]][%[[C0]], %[[C1]]]
// CHECK:         store %[[ELEM0]], %[[MEMREF]][%[[C1]], %[[C0]]]
// CHECK:         store %[[ELEM1]], %[[MEMREF]][%[[C1]], %[[C1]]]
// CHECK:         store %[[ELEM0]], %[[MEMREF]][%[[C2]], %[[C0]]]
// CHECK:         store %[[ELEM1]], %[[MEMREF]][%[[C2]], %[[C1]]]
// CHECK:         %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]]
// CHECK:         return %[[RET]] : tensor<3x2xindex>
func @tensor.from_elements_2d(%arg0: index, %arg1: index) -> tensor<3x2xindex> {
  %0 = tensor.from_elements %arg0, %arg1, %arg0, %arg1, %arg0, %arg1
         : tensor<3x2xindex>
  return %0 : tensor<3x2xindex>
}

// CHECK-LABEL: func @tensor.from_elements_3d()

// CHECK-DAG: %[[F0:.*]] = arith.constant 0.0
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

// CHECK: %[[MEMREF:.*]] = memref.alloc() : memref<3x2x2xf32>

// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index

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
func @tensor.from_elements_3d() -> tensor<3x2x2xf32> {
  %f0 = arith.constant 0.0 : f32
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
// CHECK:           %[[CASTED:.*]] = bufferization.to_memref %[[ARG]] : memref<*xf32>
// CHECK:           %[[MEMREF:.*]] = memref.alloc(%[[DYNAMIC_EXTENT]]) : memref<?xindex>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[DYNAMIC_EXTENT]]) step (%[[C1]]) {
// CHECK:             %[[ELEM:.*]] = memref.dim %[[CASTED]], %[[I]] : memref<*xf32>
// CHECK:             store %[[ELEM]], %[[MEMREF]][%[[I]]] : memref<?xindex>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]] : memref<?xindex>
// CHECK:           return %[[RET]] : tensor<?xindex>
// CHECK:         }
func @tensor.generate(%arg: tensor<*xf32>, %dynamic_extent: index) -> tensor<?xindex> {
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
// CHECK-SAME:                                                          %[[DYNAMIC_EXTENT:.*]]: index) -> tensor<16x?xindex> {
// CHECK:           %[[MEMREF:.*]] = memref.alloc(%[[DYNAMIC_EXTENT]]) : memref<16x?xindex>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[C16]], %[[DYNAMIC_EXTENT]]) step (%[[C1]], %[[C1]]) {
// CHECK:             %[[VAL_7:.*]] = arith.addi %[[I]], %[[J]] : index
// CHECK:             store %[[VAL_7]], %[[MEMREF]][%[[I]], %[[J]]] : memref<16x?xindex>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[RET:.*]] = bufferization.to_tensor %[[MEMREF]] : memref<16x?xindex>
// CHECK:           return %[[RET]] : tensor<16x?xindex>
// CHECK:         }
func @tensor.generate_static_and_dynamic(%arg0: index) -> tensor<16x?xindex> {
  %result = tensor.generate %arg0 {
  ^bb0(%i: index, %j: index):
    %sum = arith.addi %i, %j : index
    tensor.yield %sum : index
  } : tensor<16x?xindex>
  return %result : tensor<16x?xindex>
}

// The tensor.generate op needs to put its body into the
// resulting scf.parallel. To handle unknown ops in the body, it cannot clone
// the body because that would require the cloned ops to be legalized
// immediately, which is usually not possible since they might be from various
// other dialects.
//
// CHECK-LABEL: func @tensor.generate_unknown_ops_in_body
func @tensor.generate_unknown_ops_in_body(%arg0: index) -> tensor<?xindex> {
  // CHECK-NOT: tensor.generate
  %tensor = tensor.generate %arg0 {
  ^bb0(%iv: index):
    // CHECK: test.source
    %0 = "test.source"() : () -> index
    tensor.yield %0 : index
  } : tensor<?xindex>
  return %tensor : tensor<?xindex>
}
