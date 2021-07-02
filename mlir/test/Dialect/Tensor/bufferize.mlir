// RUN: mlir-opt %s -tensor-bufferize | FileCheck %s

// CHECK-LABEL:   func @dim(
// CHECK-SAME:              %[[TENSOR:.*]]: tensor<f32>,
// CHECK-SAME:              %[[INDEX:.*]]: index) -> index {
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<f32>
// CHECK:           %[[EXTENT:.*]] = memref.dim %[[MEMREF]], %[[INDEX]] : memref<f32>
// CHECK:           return %[[EXTENT]] : index
func @dim(%arg0: tensor<f32>, %arg1: index) -> index {
  %0 = tensor.dim %arg0, %arg1 : tensor<f32>
  return %0 : index
}

// CHECK-LABEL:   func @tensor.cast(
// CHECK-SAME:                      %[[TENSOR:.*]]: tensor<?xindex>) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]]
// CHECK:           %[[CASTED:.*]] = memref.cast %[[MEMREF]] : memref<?xindex> to memref<2xindex>
// CHECK:           %[[RET:.*]] = memref.tensor_load %[[CASTED]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor.cast(%arg0: tensor<?xindex>) -> tensor<2xindex> {
  %0 = tensor.cast %arg0 : tensor<?xindex> to tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL:   func @tensor.cast_from_unranked(
// CHECK-SAME:                                    %[[TENSOR:.*]]: tensor<*xf32>) -> tensor<2xf32> {
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<*xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref.cast %[[MEMREF]] : memref<*xf32> to memref<2xf32>
// CHECK:           %[[RET:.*]] = memref.tensor_load %[[CASTED_MEMREF]] : memref<2xf32>
// CHECK:           return %[[RET]] : tensor<2xf32>
func @tensor.cast_from_unranked(%arg0: tensor<*xf32>) -> tensor<2xf32> {
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL:   func @tensor.cast_to_unranked(
// CHECK-SAME:                                  %[[TENSOR:.*]]: tensor<2xf32>) -> tensor<*xf32> {
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<2xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref.cast %[[MEMREF]] : memref<2xf32> to memref<*xf32>
// CHECK:           %[[RET:.*]] = memref.tensor_load %[[CASTED_MEMREF]] : memref<*xf32>
// CHECK:           return %[[RET]] : tensor<*xf32>
func @tensor.cast_to_unranked(%arg0: tensor<2xf32>) -> tensor<*xf32> {
  %0 = tensor.cast %arg0 : tensor<2xf32> to tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL:   func @tensor.extract(
// CHECK-SAME:                  %[[TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:                  %[[IDX:.*]]: index) -> f32 {
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<?xf32>
// CHECK:           %[[RET:.*]] = memref.load %[[MEMREF]][%[[IDX]]] : memref<?xf32>
// CHECK:           return %[[RET]] : f32
// CHECK:         }
func @tensor.extract(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
  %0 = tensor.extract %arg0[%arg1] : tensor<?xf32>
  return %0 : f32
}

// CHECK-LABEL:   func @tensor.from_elements(
// CHECK-SAME:                               %[[ELEM0:.*]]: index,
// CHECK-SAME:                               %[[ELEM1:.*]]: index) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = memref.alloc()
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           store %[[ELEM0]], %[[MEMREF]][%[[C0]]]
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           store %[[ELEM1]], %[[MEMREF]][%[[C1]]]
// CHECK:           %[[RET:.*]] = memref.tensor_load %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor.from_elements(%arg0: index, %arg1: index) -> tensor<2xindex> {
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL:   func @tensor.generate(
// CHECK-SAME:                                       %[[ARG:.*]]: tensor<*xf32>,
// CHECK-SAME:                                       %[[DYNAMIC_EXTENT:.*]]: index) -> tensor<?xindex> {
// CHECK:           %[[MEMREF:.*]] = memref.alloc(%[[DYNAMIC_EXTENT]]) : memref<?xindex>
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[DYNAMIC_EXTENT]]) step (%[[C1]]) {
// CHECK:             %[[CASTED:.*]] = memref.buffer_cast %[[ARG]] : memref<*xf32>
// CHECK:             %[[ELEM:.*]] = memref.dim %[[CASTED]], %[[I]] : memref<*xf32>
// CHECK:             store %[[ELEM]], %[[MEMREF]][%[[I]]] : memref<?xindex>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[RET:.*]] = memref.tensor_load %[[MEMREF]] : memref<?xindex>
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
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[C16:.*]] = constant 16 : index
// CHECK:           scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[C16]], %[[DYNAMIC_EXTENT]]) step (%[[C1]], %[[C1]]) {
// CHECK:             %[[VAL_7:.*]] = addi %[[I]], %[[J]] : index
// CHECK:             store %[[VAL_7]], %[[MEMREF]][%[[I]], %[[J]]] : memref<16x?xindex>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[RET:.*]] = memref.tensor_load %[[MEMREF]] : memref<16x?xindex>
// CHECK:           return %[[RET]] : tensor<16x?xindex>
// CHECK:         }
func @tensor.generate_static_and_dynamic(%arg0: index) -> tensor<16x?xindex> {
  %result = tensor.generate %arg0 {
  ^bb0(%i: index, %j: index):
    %sum = addi %i, %j : index
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
