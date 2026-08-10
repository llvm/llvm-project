// RUN: mlir-opt %s -one-shot-bufferize -split-input-file | FileCheck %s

// CHECK-LABEL: memref.global "private" @global 
ml_program.global private mutable @global(dense<0> : tensor<i64>) : tensor<i64>

// CHECK-LABEL: func.func @global_load_store
func.func @global_load_store() -> i64 {
  // CHECK-DAG: %[[CST127:.*]] = arith.constant 127
  // CHECK-DAG: %[[GLOBAL_1:.*]] = memref.get_global @global
  // CHECK:     %[[VALUE:.*]] = memref.load %[[GLOBAL_1]][]
  // CHECK:     %[[NEW_VALUE:.*]] = arith.muli %[[VALUE]], %[[CST127]]
  // CHECK:     %[[ALLOC:.*]] = memref.alloc()
  // CHECK:     memref.copy %[[GLOBAL_1]], %[[ALLOC]]
  // CHECK:     memref.store %[[NEW_VALUE]], %[[ALLOC]][]
  // CHECK:     %[[GLOBAL_2:.*]] = memref.get_global @global
  // CHECK:     memref.copy %[[ALLOC]], %[[GLOBAL_2]]
  // CHECK:     return %[[NEW_VALUE]]
  %c127 = arith.constant 127 : i64
  %0 = ml_program.global_load @global : tensor<i64>
  %extracted = tensor.extract %0[] : tensor<i64>
  %1 = arith.muli %extracted, %c127 : i64
  %inserted = tensor.insert %1 into %0[] : tensor<i64>
  ml_program.global_store @global = %inserted : tensor<i64>
  return %1 : i64
}

// -----

// CHECK-LABEL: memref.global "private" @global 
ml_program.global private mutable @global(dense<0> : tensor<i64>) : tensor<i64>

// CHECK-LABEL: func.func @raw_hazard
func.func @raw_hazard() -> i64 {
  // CHECK-DAG: %[[CST127:.*]] = arith.constant 127
  // CHECK-DAG: %[[GLOBAL_1:.*]] = memref.get_global @global
  // CHECK-DAG: %[[GLOBAL_2:.*]] = memref.get_global @global
  // CHECK-DAG: %[[ALLOC:.*]] = memref.alloc()
  // CHECK:     memref.copy %[[GLOBAL_1]], %[[ALLOC]]
  // CHECK:     memref.store %[[CST127]], %[[ALLOC]][]
  // CHECK:     %[[VAL:.*]] = memref.load %[[GLOBAL_2]][]
  // CHECK:     %[[GLOBAL_3:.*]] = memref.get_global @global
  // CHECK:     memref.copy %[[ALLOC]], %[[GLOBAL_3]]
  // CHECK:     return %[[VAL]]
  %c127 = arith.constant 127 : i64
  %0 = ml_program.global_load @global : tensor<i64>
  %1 = ml_program.global_load @global : tensor<i64>
  %inserted = tensor.insert %c127 into %0[] : tensor<i64>
  %extracted = tensor.extract %1[] : tensor<i64>
  ml_program.global_store @global = %inserted : tensor<i64>
  return %extracted : i64
}

// -----

// CHECK-LABEL: memref.global "private" @state_tensor
ml_program.global private mutable @"state_tensor"(dense<0.0> : tensor<4x75xf32>) : tensor<4x75xf32>

// CHECK-LABEL: func.func @global_load_store_tensor
func.func @global_load_store_tensor() -> tensor<4x75xf32> {
  // CHECK-DAG:     %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:     %[[CST:.*]] = arith.constant 1.000000e+00
  // CHECK-DAG:     %[[GLOB:.*]] = memref.get_global @state_tensor
  // CHECK:         %[[VAL:.*]] = memref.load %[[GLOB]][%[[C0]], %[[C0]]]
  // CHECK:         %[[ADD:.*]] = arith.addf %[[VAL]], %[[CST]]
  // CHECK:         %[[ALLOC1:.*]] = memref.alloc() {alignment = 64 : i64}
  // CHECK:         memref.copy %[[GLOB]], %[[ALLOC1]] 
  // CHECK:         memref.store %[[ADD]], %[[ALLOC1]][%[[C0]], %[[C0]]] 
  // CHECK:         %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC1]] 
  // CHECK:         %[[ALLOC2:.*]] = memref.alloc() {alignment = 64 : i64}
  // CHECK:         memref.copy %[[ALLOC1]], %[[ALLOC2]] 
  // CHECK:         %[[GLOB_REF:.*]] = memref.get_global @state_tensor 
  // CHECK:         memref.copy %[[ALLOC2]], %[[GLOB_REF]] 
  // CHECK:         return %[[TENSOR]]
  %c0 = arith.constant 0 : index
  %cst_val = arith.constant 1.0 : f32
  %initial_state = ml_program.global_load @"state_tensor" : tensor<4x75xf32>
  %val = tensor.extract %initial_state[%c0, %c0] : tensor<4x75xf32>
  %next_val = arith.addf %val, %cst_val : f32
  %updated_tensor = tensor.insert %next_val into %initial_state[%c0, %c0] : tensor<4x75xf32>
  ml_program.global_store @"state_tensor" = %updated_tensor : tensor<4x75xf32>
  return %updated_tensor : tensor<4x75xf32>
}

