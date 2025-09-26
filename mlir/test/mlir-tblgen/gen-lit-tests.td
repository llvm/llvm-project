// RUN: mlir-tblgen -gen-lit-tests -I %S/../../include -dialect=test %s | FileCheck %s

include "mlir/Pass/PassBase.td"
include "mlir/IR/Testable.td"

def TestPassWithEmbeddedLitTests : Pass<"test-pass-with-embedded-lit-tests"> {
  let summary = "pass summary";
  let description = [{
    Pass description
  }];
  
  let tests = [
    LitTest<
      "lit_test_file_1.mlir", 
      [{
          func.func @test1() {
            return 42;
          }
      }],
      [
        "// RUN: mlir-opt %s --verify-roundtrip | FileCheck %s",
      ],
      [
        "// RANDOM-CHECK-LABEL: func.func @test1",
      ]
    >,
    LitTest<
      "lit_test_file_2.mlir", 
      [{
          func.func @test2() {
            return 42;
          }
      }],
      [
        "// RUN: mlir-opt %s --verify-roundtrip | FileCheck %s",
      ],
      [
        "// RANDOM-CHECK-LABEL: func.func @test2",
      ]
    >,
  ];
}

// CHECK-LABEL:       // Generated 2 LIT test files
// CHECK:             // Use the following files for LIT testing:

// CHECK:             // File: generated_TestPassWithEmbeddedLitTests_lit_test_file_1.mlir
// CHECK:             // --- BEGIN generated_TestPassWithEmbeddedLitTests_lit_test_file_1.mlir ---
// CHECK:             // RUN: mlir-opt %s --verify-roundtrip | FileCheck %s
// CHECK:             // Generated from TableGen definition: TestPassWithEmbeddedLitTests
// CHECK:             func.func @test1() {
// CHECK:                return 42;
// CHECK:             }
// CHECK:             // RANDOM-CHECK-LABEL: func.func @test1
// CHECK:             --- END generated_TestPassWithEmbeddedLitTests_lit_test_file_1.mlir ---

// CHECK:             // File: generated_TestPassWithEmbeddedLitTests_lit_test_file_2.mlir
// CHECK:             // --- BEGIN generated_TestPassWithEmbeddedLitTests_lit_test_file_2.mlir ---
// CHECK:             // RUN: mlir-opt %s --verify-roundtrip | FileCheck %s
// CHECK:             // Generated from TableGen definition: TestPassWithEmbeddedLitTests
// CHECK:             func.func @test2() {
// CHECK:               return 42;
// CHECK:             }
// CHECK:             // RANDOM-CHECK-LABEL: func.func @test2
// CHECK:             // --- END generated_TestPassWithEmbeddedLitTests_lit_test_file_2.mlir ---