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

