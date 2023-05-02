// RUN: mlir-opt %s -test-return-type -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: testCreateFunctions
// This function tests invoking the create method with different inference
// methods. The attributes of the ops inside are used to test creation.
func.func @testCreateFunctions(%arg0 : tensor<10xf32, !test.smpla>, %arg1 : tensor<20xi32>) {
// CHECK: "test.no_attributes"
  %good = "test.no_attributes"(%arg0, %arg0) : (tensor<10xf32, !test.smpla>, tensor<10xf32, !test.smpla>) -> tensor<10xf32, !test.smpla>
// CHECK: "test.op_with_shaped_type_infer_type_if"
// CHECK-SAME: (tensor<10xf32, !test.smpla>, tensor<10xf32, !test.smpla>) -> tensor<10xi17, !test.smpla>
// CHECK: "test.op_with_shaped_type_infer_type_if"
// CHECK-SAME: (tensor<10xf32, !test.smpla>, tensor<20xi32>) -> tensor<10xi17, !test.smpla>
// CHECK: "test.op_with_shaped_type_infer_type_if"
// CHECK-SAME: (tensor<20xi32>, tensor<10xf32, !test.smpla>) -> tensor<20xi17>
// CHECK: "test.op_with_shaped_type_infer_type_if"
// CHECK-SAME: (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi17>
// CHECK: "test.op_with_infer_type_if"
// CHECK-SAME: (tensor<10xf32, !test.smpla>, tensor<10xf32, !test.smpla>) -> tensor<10xf32, !test.smpla>
// CHECK: "test.op_with_infer_type_if"
// CHECK-SAME: (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi32>
  return
}

// -----

func.func @testReturnTypeOpInterface(%arg0 : tensor<10xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{incompatible with return type}}
  %bad = "test.op_with_infer_type_if"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<*xf32>
  return
}

// -----

func.func @testReturnTypeOpInterfaceMismatch(%arg0 : tensor<10xf32>, %arg1 : tensor<20xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand type mismatch}}
  %bad = "test.op_with_infer_type_if"(%arg0, %arg1) : (tensor<10xf32>, tensor<20xf32>) -> tensor<*xf32>
  return
}

// -----

func.func @testReturnTypeOpInterface(%arg0 : tensor<10xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{required first operand and result to match}}
  %bad = "test.op_with_refine_type_if"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<*xf32>
  return
}

// -----
// CHECK-LABEL: testReifyFunctions
func.func @testReifyFunctions(%arg0 : tensor<10xf32>, %arg1 : tensor<20xf32>) {
  // expected-remark@+1 {{arith.constant 10}}
  %0 = "test.op_with_shaped_type_infer_type_if"(%arg0, %arg1) : (tensor<10xf32>, tensor<20xf32>) -> tensor<10xi17>
  // expected-remark@+1 {{arith.constant 20}}
  %1 = "test.op_with_shaped_type_infer_type_if"(%arg1, %arg0) : (tensor<20xf32>, tensor<10xf32>) -> tensor<20xi17>
  return
}
