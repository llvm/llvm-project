// RUN: mlir-opt %s -test-one-shot-module-bufferize="allow-return-allocs-from-loops test-tensor-to-memref" -split-input-file | FileCheck %s --check-prefixes=CHECK,STATIC
// RUN: mlir-opt %s -test-one-shot-module-bufferize="allow-return-allocs-from-loops test-tensor-to-dynamic-memref" -split-input-file -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,DYNAMIC

// The test tensor type bufferizes to a memref with static extents under
// test-tensor-to-memref and with dynamic extents under
// test-tensor-to-dynamic-memref. A copy of a value takes the shape of the
// buffer it copies; an allocation without a copy takes the shape of the
// result's buffer type, so it needs no sizes when that shape is static and
// is refused when it is dynamic.

// CHECK-LABEL: func.func @custom_types_scf_for_yield_outside_value(
// STATIC-SAME:    %[[a:[^:]+]]: memref<4xf64>, %[[b:[^:]+]]: memref<4xf64>
// DYNAMIC-SAME:   %[[a:[^:]+]]: memref<?xf64>, %[[b:[^:]+]]: memref<?xf64>
// CHECK: scf.for
// STATIC:   %[[alloc:.+]] = memref.alloc() {{.*}}: memref<4xf64>
// DYNAMIC:  %[[dim:.+]] = memref.dim %[[b]]
// DYNAMIC:  %[[alloc:.+]] = memref.alloc(%[[dim]]) {{.*}}: memref<?xf64>
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
// STATIC-SAME:    %[[arg:[^:]+]]: memref<4xf64>
// DYNAMIC-SAME:   %[[arg:[^:]+]]: memref<?xf64>
// STATIC:  %[[alloc:.+]] = memref.alloc() {{.*}}: memref<4xf64>
// DYNAMIC: %[[dim:.+]] = memref.dim %[[arg]]
// DYNAMIC: %[[alloc:.+]] = memref.alloc(%[[dim]]) {{.*}}: memref<?xf64>
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

// A copy of a value whose buffer has static extents takes that static shape
// under either conversion.

// CHECK-LABEL: func.func @custom_types_copy_static_buffer(
// CHECK-SAME:    %[[m:[^:]+]]: memref<4xf64>
// CHECK: %[[alloc:.+]] = memref.alloc() {{.*}}: memref<4xf64>
// CHECK: memref.copy %[[m]], %[[alloc]]
func.func @custom_types_copy_static_buffer(%m: memref<4xf64>)
    -> (!test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>) {
  %t = bufferization.to_tensor %m restrict writable
    : memref<4xf64> to !test.test_tensor<[4], f64>
  %0 = "test.dummy_tensor_op"(%t)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  %1 = "test.dummy_tensor_op"(%t)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  return %0, %1 : !test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>
}

// -----

// An allocation without a copy takes the shape of the result's buffer type:
// a static shape needs no sizes; a dynamic one has none to take.

// STATIC-LABEL: func.func @custom_types_operand_alloc(
// STATIC-NOT: memref.copy
// STATIC: memref.alloc() {{.*}}: memref<4xf64>
// STATIC-NOT: memref.copy
// STATIC: memref.alloc() {{.*}}: memref<4xf64>
// STATIC-NOT: memref.copy
// STATIC: return
func.func @custom_types_operand_alloc()
    -> (!test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>) {
  // expected-error @+2 {{allocating 'memref<?xf64>' needs a size for each dynamic dimension, which the op takes only for a ranked tensor result}}
  // expected-error @+1 {{failed to bufferize op}}
  %e = bufferization.alloc_tensor() : !test.test_tensor<[4], f64>
  %0 = "test.dummy_tensor_op"(%e)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  %1 = "test.dummy_tensor_op"(%e)
    : (!test.test_tensor<[4], f64>) -> !test.test_tensor<[4], f64>
  return %0, %1 : !test.test_tensor<[4], f64>, !test.test_tensor<[4], f64>
}
