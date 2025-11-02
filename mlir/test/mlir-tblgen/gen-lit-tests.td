// RUN: mlir-tblgen -gen-lit-tests -I %S/../../include -dialect=test %s | FileCheck %s

include "mlir/Pass/PassBase.td"
include "mlir/IR/Testable.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "test";
}

def TestOp : Op<Test_Dialect, "test_op"> {
  let summary = "test op with mlir_example code blocks";
  let description = [{
    This operation demonstrates the mlir_example feature for ops.

    Basic usage:
    ```mlir_example
    func.func @foo(%arg0: i32) -> i32 {
      %0 = test.test_op %arg0 : i32
      return %0 : i32
    }
    ```

    And some more examples -

    ```mlir_example
    func.func @foo1(%arg1: i32) -> i32 {
      %0 = test.test_op %arg1 : i32
      return %0 : i32
    }
    ```
  }];

  let arguments = (ins I32:$input);
  let results = (outs I32:$output);
}

// CHECK:             // File: generated_TestOp_example_0.mlir
// CHECK:             // --- BEGIN generated_TestOp_example_0.mlir ---
// CHECK:             // RUN: mlir-opt %s --verify-roundtrip
// CHECK:             // Generated from TableGen definition: TestOp
// CHECK:             func.func @foo(%arg0: i32) -> i32 {
// CHECK:               %0 = test.test_op %arg0 : i32
// CHECK:               return %0 : i32
// CHECK:             }
// CHECK:             // --- END generated_TestOp_example_0.mlir ---

// CHECK:             // File: generated_TestOp_example_1.mlir
// CHECK:             // --- BEGIN generated_TestOp_example_1.mlir ---
// CHECK:             // RUN: mlir-opt %s --verify-roundtrip
// CHECK:             // Generated from TableGen definition: TestOp
// CHECK:             func.func @bar(%arg0: i32, %arg1: i32) -> i32 {
// CHECK:               %0 = test.test_op %arg0 : i32
// CHECK:               %1 = test.test_op %arg1 : i32
// CHECK:               %2 = arith.addi %0, %1 : i32
// CHECK:               return %2 : i32
// CHECK:             }
// CHECK:             // --- END generated_TestOp_example_1.mlir ---