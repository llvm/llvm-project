// RUN: mlir-opt %s --test-bytecode-roundtrip="test-dialect-version=1.2" -verify-diagnostics | FileCheck %s --check-prefix=VERSION_1_2
// RUN: mlir-opt %s --test-bytecode-roundtrip="test-dialect-version=2.0" -verify-diagnostics | FileCheck %s --check-prefix=VERSION_2_0
// RUN: mlir-opt %s --split-input-file --test-bytecode-roundtrip="test-dialect-version=2.0" -verify-diagnostics | FileCheck %s --check-prefix=NO_TEST_DIALECT

func.func @base_test(%arg0 : i32) -> f32 {
  %0 = "test.addi"(%arg0, %arg0) : (i32, i32) -> i32
  %1 = "test.cast"(%0) : (i32) -> f32
  return %1 : f32
}

// VERSION_1_2: Overriding IntegerType encoding...
// VERSION_1_2: Overriding parsing of IntegerType encoding...

// VERSION_2_0-NOT: Overriding IntegerType encoding...
// VERSION_2_0-NOT: Overriding parsing of IntegerType encoding...

// -----

// Regression test for https://github.com/llvm/llvm-project/issues/128321:
// test-bytecode-roundtrip must not crash when the test dialect is absent from
// the module (i.e., no test dialect types are present in the bytecode).

// NO_TEST_DIALECT-NOT: Overriding IntegerType encoding...
// NO_TEST_DIALECT: func.func @no_test_dialect_ops
func.func @no_test_dialect_ops() {
  return
}
