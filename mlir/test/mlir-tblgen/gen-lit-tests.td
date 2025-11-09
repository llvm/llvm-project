// RUN: mlir-tblgen -gen-lit-tests -I %S/../../include %s | FileCheck %s

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
    ```mlir_example(mlir-opt)
    # func.func @foo(%arg0: i32) -> i32 {
      %0 = test.test_op %arg0 : i32
      return %0 : i32
    # }
    ```

    And some more back to back examples -

    ```mlir_example(some-other-tool)
    # func.func @foo1(%arg1: i32) -> i32 {
      %0 = test.test_op %arg1 : i32
      return %0 : i32
    # }
    ```
    ```mlir_example(yet-another-tool)
    # func.func @foo2(%arg2: i32) -> i32 {
      %0 = test.test_op %arg2 : i32
      return %0 : i32
    # }
    ```
  }];

  let arguments = (ins I32:$input);
  let results = (outs I32:$output);
}

// CHECK-LABEL:     // Generated 3 LIT test files
// CHECK:           // Use the following files for LIT testing:

// CHECK:           // File: generated_TestOp_example_0.mlir
// CHECK:           // --- BEGIN generated_TestOp_example_0.mlir ---
// CHECK:           mlir-opt %s --verify-roundtrip
// CHECK:           // Generated from TableGen definition: TestOp
// CHECK:           func.func @foo(%arg0: i32) -> i32 {
// CHECK:           %0 = test.test_op %arg0 : i32
// CHECK:           return %0 : i32
// CHECK:           }
// CHECK:           // --- END generated_TestOp_example_0.mlir ---

// CHECK:           // File: generated_TestOp_example_1.mlir
// CHECK:           // --- BEGIN generated_TestOp_example_1.mlir ---
// CHECK:           some-other-tool %s --verify-roundtrip
// CHECK:           // Generated from TableGen definition: TestOp
// CHECK:           func.func @foo1(%arg1: i32) -> i32 {
// CHECK:           %0 = test.test_op %arg1 : i32
// CHECK:           return %0 : i32
// CHECK:           }
// CHECK:           // --- END generated_TestOp_example_1.mlir ---

// CHECK:           // File: generated_TestOp_example_2.mlir
// CHECK:           // --- BEGIN generated_TestOp_example_2.mlir ---
// CHECK:           yet-another-tool %s --verify-roundtrip
// CHECK:           // Generated from TableGen definition: TestOp
// CHECK:           func.func @foo2(%arg2: i32) -> i32 {
// CHECK:           %0 = test.test_op %arg2 : i32
// CHECK:           return %0 : i32
// CHECK:           }
// CHECK:           // --- END generated_TestOp_example_2.mlir ---
