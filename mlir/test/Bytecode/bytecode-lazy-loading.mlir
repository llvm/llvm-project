// RUN: mlir-opt --pass-pipeline="builtin.module(test-lazy-loading)" %s -o %t | FileCheck %s
// RUN: mlir-opt --pass-pipeline="builtin.module(test-lazy-loading{bytecode-version=1})" %s -o %t | FileCheck %s --check-prefix=OLD-BYTECODE


func.func @op_with_passthrough_region_args() {
  %0 = arith.constant 10 : index

  // Ensure we can handle nested non-isolated/non-lazy regions.
  "test.one_region_op"() ({
    "test.consumer"(%0) : (index) -> ()
  }) : () -> ()

  test.isolated_region %0 {
    "test.consumer"(%0) : (index) -> ()
  }
  %result:2 = "test.op"() : () -> (index, index)
  test.isolated_region %result#1 {
    "test.consumer"(%result#1) : (index) -> ()
  }
  
  test.isolated_regions {
    "test.unknown_op"() : () -> ()
  }, {
    "test.unknown_op"() : () -> ()
  }
  return
}

// Before version 2, we can't support lazy loading.
// OLD-BYTECODE-NOT: Has 1 ops to materialize
// OLD-BYTECODE-NOT: Materializing
// OLD-BYTECODE: Has 0 ops to materialize


// CHECK: Has 1 ops to materialize

// CHECK: Before Materializing...
// CHECK: "builtin.module"() ({
// CHECK-NOT: func
// CHECK: Materializing...
// CHECK: "builtin.module"() ({
// CHECK: "func.func"() <{function_type = () -> (), sym_name = "op_with_passthrough_region_args"}> ({
// CHECK-NOT: arith
// CHECK: Has 1 ops to materialize

// CHECK: Before Materializing...
// CHECK: "func.func"() <{function_type = () -> (), sym_name = "op_with_passthrough_region_args"}> ({
// CHECK-NOT: arith
// CHECK: Materializing...
// CHECK: "func.func"() <{function_type = () -> (), sym_name = "op_with_passthrough_region_args"}> ({
// CHECK: arith
// CHECK: one_region_op
// CHECK: test.consumer
// CHECK: isolated_region
// CHECK-NOT: test.consumer
// CHECK: Has 3 ops to materialize

// CHECK: Before Materializing...
// CHECK: test.isolated_region
// CHECK-NOT:  test.consumer
// CHECK: Materializing...
// CHECK: test.isolated_region
// CHECK: ^bb0(%arg0: index):
// CHECK:  test.consumer
// CHECK: Has 2 ops to materialize

// CHECK: Before Materializing...
// CHECK: test.isolated_region
// CHECK-NOT: test.consumer
// CHECK: Materializing...
// CHECK: test.isolated_region
// CHECK: test.consumer
// CHECK: Has 1 ops to materialize

// CHECK: Before Materializing...
// CHECK: test.isolated_regions
// CHECK-NOT: test.unknown_op
// CHECK: Materializing...
// CHECK: test.isolated_regions
// CHECK: test.unknown_op
// CHECK: test.unknown_op
// CHECK: Has 0 ops to materialize
