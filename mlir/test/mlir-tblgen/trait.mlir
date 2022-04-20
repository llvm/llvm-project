// RUN: mlir-opt -test-trait-folder %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test that involutions fold correctly
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @testSingleInvolution
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testSingleInvolution(%arg0 : i32) -> i32 {
  // CHECK: [[INVOLUTION:%.+]] = "test.op_involution_trait_no_operation_fold"([[ARG0]])
  %0 = "test.op_involution_trait_no_operation_fold"(%arg0) : (i32) -> i32
  // CHECK: return [[INVOLUTION]]
  return %0: i32
}

// CHECK-LABEL: func @testDoubleInvolution
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testDoubleInvolution(%arg0: i32) -> i32 {
  %0 = "test.op_involution_trait_no_operation_fold"(%arg0) : (i32) -> i32
  %1 = "test.op_involution_trait_no_operation_fold"(%0) : (i32) -> i32
  // CHECK: return [[ARG0]]
  return %1: i32
}

// CHECK-LABEL: func @testTripleInvolution
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testTripleInvolution(%arg0: i32) -> i32 {
  // CHECK: [[INVOLUTION:%.+]] = "test.op_involution_trait_no_operation_fold"([[ARG0]])
  %0 = "test.op_involution_trait_no_operation_fold"(%arg0) : (i32) -> i32
  %1 = "test.op_involution_trait_no_operation_fold"(%0) : (i32) -> i32
  %2 = "test.op_involution_trait_no_operation_fold"(%1) : (i32) -> i32
  // CHECK: return [[INVOLUTION]]
  return %2: i32
}

//===----------------------------------------------------------------------===//
// Test that involutions fold occurs if operation fold fails
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @testFailingOperationFolder
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testFailingOperationFolder(%arg0: i32) -> i32 {
  %0 = "test.op_involution_trait_failing_operation_fold"(%arg0) : (i32) -> i32
  %1 = "test.op_involution_trait_failing_operation_fold"(%0) : (i32) -> i32
  // CHECK: return [[ARG0]]
  return %1: i32
}

//===----------------------------------------------------------------------===//
// Test that involution fold does not occur if operation fold succeeds
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @testInhibitInvolution
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testInhibitInvolution(%arg0: i32) -> i32 {
  // CHECK: [[OP:%.+]] = "test.op_involution_trait_succesful_operation_fold"([[ARG0]])
  %0 = "test.op_involution_trait_succesful_operation_fold"(%arg0) : (i32) -> i32
  %1 = "test.op_involution_trait_succesful_operation_fold"(%0) : (i32) -> i32
  // CHECK: return [[OP]]
  return %1: i32
}

//===----------------------------------------------------------------------===//
// Test that idempotent folding works correctly
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @testSingleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testSingleIdempotent(%arg0 : i32) -> i32 {
  // CHECK: [[IDEMPOTENT:%.+]] = "test.op_idempotent_trait"([[ARG0]])
  %0 = "test.op_idempotent_trait"(%arg0) : (i32) -> i32
  // CHECK: return [[IDEMPOTENT]]
  return %0: i32
}

// CHECK-LABEL: func @testDoubleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testDoubleIdempotent(%arg0: i32) -> i32 {
  // CHECK: [[IDEMPOTENT:%.+]] = "test.op_idempotent_trait"([[ARG0]])
  %0 = "test.op_idempotent_trait"(%arg0) : (i32) -> i32
  %1 = "test.op_idempotent_trait"(%0) : (i32) -> i32
  // CHECK: return [[IDEMPOTENT]]
  return %1: i32
}

// CHECK-LABEL: func @testTripleIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testTripleIdempotent(%arg0: i32) -> i32 {
  // CHECK: [[IDEMPOTENT:%.+]] = "test.op_idempotent_trait"([[ARG0]])
  %0 = "test.op_idempotent_trait"(%arg0) : (i32) -> i32
  %1 = "test.op_idempotent_trait"(%0) : (i32) -> i32
  %2 = "test.op_idempotent_trait"(%1) : (i32) -> i32
  // CHECK: return [[IDEMPOTENT]]
  return %2: i32
}

// CHECK-LABEL: func @testBinaryIdempotent
// CHECK-SAME:  ([[ARG0:%.+]]: i32)
func.func @testBinaryIdempotent(%arg0 : i32) -> i32 {
  %0 = "test.op_idempotent_trait_binary"(%arg0, %arg0) : (i32, i32) -> i32
  // CHECK: return [[ARG0]]
  return %0: i32
}
