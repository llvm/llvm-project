// RUN: mlir-opt %s -test-one-shot-module-bufferize="allow-return-allocs-from-loops test-tensor-to-memref" -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @custom_types_scf_for_yield_outside_value(
// CHECK-SAME:    %[[a:[^:]+]]: memref<4xf64>, %[[b:[^:]+]]: memref<4xf64>
// CHECK: scf.for
// CHECK:   %[[alloc:.+]] = memref.alloc() {{.*}}: memref<4xf64>
// CHECK:   memref.copy %[[b]], %[[alloc]]
// CHECK:   scf.yield %[[alloc]]
func.func @custom_types_scf_for_yield_outside_value(
    %a: !test.test_tensor<[4], f64>, %b: !test.test_tensor<[4], f64>,
    %lb: index, %ub: index, %step: index) -> !test.test_tensor<[4], f64> {
  %loop = scf.for %i = %lb to %ub step %step
      iter_args(%iter = %a) -> (!test.test_tensor<[4], f64>) {
    scf.yield %b : !test.test_tensor<[4], f64>
  }
  return %loop : !test.test_tensor<[4], f64>
}

// -----

// CHECK-LABEL: func.func @custom_types_operand_copy(
// CHECK-SAME:    %[[arg:[^:]+]]: memref<4xf64>
// CHECK: %[[alloc:.+]] = memref.alloc() {{.*}}: memref<4xf64>
// CHECK: memref.copy %[[arg]], %[[alloc]]
// CHECK: "test.dummy_memref_op"(%[[alloc]])
// CHECK: "test.dummy_memref_op"(%[[arg]])
func.func @custom_types_operand_copy(%arg: !test.test_tensor<[4], f64>)
    -> (!test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>) {
  %0 = "test.dummy_tensor_op"(%arg)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  %1 = "test.dummy_tensor_op"(%arg)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  return %0, %1 : !test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>
}

// -----

// CHECK-LABEL: func.func @custom_types_operand_alloc(
// CHECK-NOT: memref.copy
// CHECK: memref.alloc() {{.*}}: memref<4xf64>
// CHECK-NOT: memref.copy
// CHECK: memref.alloc() {{.*}}: memref<4xf64>
// CHECK-NOT: memref.copy
// CHECK: return
func.func @custom_types_operand_alloc()
    -> (!test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>) {
  %e = bufferization.alloc_tensor() : !test.test_tensor<[4], f64>
  %0 = "test.dummy_tensor_op"(%e)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  %1 = "test.dummy_tensor_op"(%e)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  return %0, %1 : !test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>
}
