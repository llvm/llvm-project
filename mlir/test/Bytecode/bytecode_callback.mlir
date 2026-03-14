// RUN: mlir-opt %s --test-bytecode-roundtrip="test-dialect-version=1.2" -verify-diagnostics | FileCheck %s --check-prefix=VERSION_1_2
// RUN: mlir-opt %s --test-bytecode-roundtrip="test-dialect-version=2.0" -verify-diagnostics | FileCheck %s --check-prefix=VERSION_2_0

func.func @base_test(%arg0 : i32) -> f32 {
  %0 = "test.addi"(%arg0, %arg0) : (i32, i32) -> i32
  %1 = "test.cast"(%0) : (i32) -> f32
  return %1 : f32
}

// VERSION_1_2: Overriding IntegerType encoding...
// VERSION_1_2: Overriding parsing of IntegerType encoding...

// VERSION_2_0-NOT: Overriding IntegerType encoding...
// VERSION_2_0-NOT: Overriding parsing of IntegerType encoding...
