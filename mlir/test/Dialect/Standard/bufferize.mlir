// RUN: mlir-opt %s -std-bufferize | FileCheck %s

// CHECK-LABEL:   func @dynamic_tensor_from_elements(
// CHECK-SAME:                                       %[[ARG:.*]]: tensor<*xf32>,
// CHECK-SAME:                                       %[[DYNAMIC_EXTENT:.*]]: index) -> tensor<?xindex> {
// CHECK:           %[[MEMREF:.*]] = alloc(%[[DYNAMIC_EXTENT]]) : memref<?xindex>
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[DYNAMIC_EXTENT]]) step (%[[C1]]) {
// CHECK:             %[[ELEM:.*]] = dim %[[ARG]], %[[I]] : tensor<*xf32>
// CHECK:             store %[[ELEM]], %[[MEMREF]][%[[I]]] : memref<?xindex>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[RET:.*]] = tensor_load %[[MEMREF]] : memref<?xindex>
// CHECK:           return %[[RET]] : tensor<?xindex>
// CHECK:         }
func @dynamic_tensor_from_elements(%arg: tensor<*xf32>, %rank: index) -> tensor<?xindex> {
  %result = dynamic_tensor_from_elements %rank {
  ^bb0(%i : index):
    %elem = dim %arg, %i : tensor<*xf32>
    yield %elem : index
  } : tensor<?xindex>
  return %result : tensor<?xindex>
}

// Additional test that checks the logic for intermixed static and dynamic
// extents.
//
// CHECK-LABEL:   func @dynamic_tensor_from_elements_static_and_dynamic(
// CHECK-SAME:                                                          %[[DYNAMIC_EXTENT:.*]]: index) -> tensor<16x?xindex> {
// CHECK:           %[[MEMREF:.*]] = alloc(%[[DYNAMIC_EXTENT]]) : memref<16x?xindex>
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[C16:.*]] = constant 16 : index
// CHECK:           scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[C16]], %[[DYNAMIC_EXTENT]]) step (%[[C1]], %[[C1]]) {
// CHECK:             %[[VAL_7:.*]] = addi %[[I]], %[[J]] : index
// CHECK:             store %[[VAL_7]], %[[MEMREF]][%[[I]], %[[J]]] : memref<16x?xindex>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[RET:.*]] = tensor_load %[[MEMREF]] : memref<16x?xindex>
// CHECK:           return %[[RET]] : tensor<16x?xindex>
// CHECK:         }
func @dynamic_tensor_from_elements_static_and_dynamic(%arg0: index) -> tensor<16x?xindex> {
  %result = dynamic_tensor_from_elements %arg0 {
  ^bb0(%i: index, %j: index):
    %sum = addi %i, %j : index
    yield %sum : index
  } : tensor<16x?xindex>
  return %result : tensor<16x?xindex>
}

// CHECK-LABEL:   func @extract_element(
// CHECK-SAME:                          %[[TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:                          %[[IDX:.*]]: index) -> f32 {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<?xf32>
// CHECK:           %[[RET:.*]] = load %[[MEMREF]][%[[IDX]]] : memref<?xf32>
// CHECK:           return %[[RET]] : f32
// CHECK:         }
func @extract_element(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
  %0 = extract_element %arg0[%arg1] : tensor<?xf32>
  return %0 : f32
}

// CHECK-LABEL:   func @select(
// CHECK-SAME:                 %[[PRED:.*]]: i1,
// CHECK-SAME:                 %[[TRUE_VAL:.*]]: tensor<f32>,
// CHECK-SAME:                 %[[FALSE_VAL:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[TRUE_VAL_MEMREF:.*]] = tensor_to_memref %[[TRUE_VAL]] : memref<f32>
// CHECK:           %[[FALSE_VAL_MEMREF:.*]] = tensor_to_memref %[[FALSE_VAL]] : memref<f32>
// CHECK:           %[[RET_MEMREF:.*]] = select %[[PRED]], %[[TRUE_VAL_MEMREF]], %[[FALSE_VAL_MEMREF]] : memref<f32>
// CHECK:           %[[RET:.*]] = tensor_load %[[RET_MEMREF]] : memref<f32>
// CHECK:           return %[[RET]] : tensor<f32>
func @select(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = select %arg0, %arg1, %arg2 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL:   func @tensor_cast(
// CHECK-SAME:                      %[[TENSOR:.*]]: tensor<?xindex>) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]]
// CHECK:           %[[CASTED:.*]] = memref_cast %[[MEMREF]] : memref<?xindex> to memref<2xindex>
// CHECK:           %[[RET:.*]] = tensor_load %[[CASTED]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor_cast(%arg0: tensor<?xindex>) -> tensor<2xindex> {
  %0 = tensor_cast %arg0 : tensor<?xindex> to tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL:   func @tensor_cast_from_unranked(
// CHECK-SAME:                                    %[[TENSOR:.*]]: tensor<*xf32>) -> tensor<2xf32> {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<*xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref_cast %[[MEMREF]] : memref<*xf32> to memref<2xf32>
// CHECK:           %[[RET:.*]] = tensor_load %[[CASTED_MEMREF]] : memref<2xf32>
// CHECK:           return %[[RET]] : tensor<2xf32>
func @tensor_cast_from_unranked(%arg0: tensor<*xf32>) -> tensor<2xf32> {
  %0 = tensor_cast %arg0 : tensor<*xf32> to tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL:   func @tensor_cast_to_unranked(
// CHECK-SAME:                                  %[[TENSOR:.*]]: tensor<2xf32>) -> tensor<*xf32> {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<2xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref_cast %[[MEMREF]] : memref<2xf32> to memref<*xf32>
// CHECK:           %[[RET:.*]] = tensor_load %[[CASTED_MEMREF]] : memref<*xf32>
// CHECK:           return %[[RET]] : tensor<*xf32>
func @tensor_cast_to_unranked(%arg0: tensor<2xf32>) -> tensor<*xf32> {
  %0 = tensor_cast %arg0 : tensor<2xf32> to tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL:   func @tensor_from_elements(
// CHECK-SAME:                               %[[ELEM0:.*]]: index,
// CHECK-SAME:                               %[[ELEM1:.*]]: index) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = alloc()
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           store %[[ELEM0]], %[[MEMREF]][%[[C0]]]
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           store %[[ELEM1]], %[[MEMREF]][%[[C1]]]
// CHECK:           %[[RET:.*]] = tensor_load %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor_from_elements(%arg0: index, %arg1: index) -> tensor<2xindex> {
  %0 = tensor_from_elements %arg0, %arg1 : tensor<2xindex>
  return %0 : tensor<2xindex>
}

// The dynamic_tensor_from_elements op clones each op in its body.
// Make sure that regions nested within such ops are recursively converted.
// CHECK-LABEL: func @recursively_convert_cloned_regions
func @recursively_convert_cloned_regions(%arg0: tensor<?xindex>, %arg1: index, %arg2: i1) -> tensor<?xindex> {
  %tensor = dynamic_tensor_from_elements %arg1 {
  ^bb0(%iv: index):
    %48 = scf.if %arg2 -> (index) {
      scf.yield %iv : index
    } else {
      // CHECK-NOT: extract_element
      %50 = extract_element %arg0[%iv] : tensor<?xindex>
      scf.yield %50 : index
    }
    yield %48 : index
  } : tensor<?xindex>
  return %tensor : tensor<?xindex>
}
