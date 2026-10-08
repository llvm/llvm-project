// RUN: mlir-opt %s -split-input-file --test-bytecode-roundtrip="test-kind=1" | FileCheck %s --check-prefix=TEST_1
// RUN: mlir-opt %s -split-input-file --test-bytecode-roundtrip="test-kind=2" | FileCheck %s --check-prefix=TEST_2

func.func @base_test(%arg0: !test.i32, %arg1: f32) {
  return
}

// TEST_1: Overriding TestI32Type encoding...
// TEST_1: func.func @base_test([[ARG0:%.+]]: i32, [[ARG1:%.+]]: f32) {

// -----

func.func @base_test(%arg0: i32, %arg1: f32) {
  return
}

// TEST_2: Overriding parsing of TestI32Type encoding...
// TEST_2: func.func @base_test([[ARG0:%.+]]: !test.i32, [[ARG1:%.+]]: f32) {

// -----

// Regression test: complex types such as memref must round-trip without
// crashing when the test-kind=2 type callback calls iface->readType() for
// every builtin type. Previously this crashed because the bytecode reading
// path used get<T>() (which asserts) instead of getChecked<T>() (which
// returns null on invalid input).

func.func @test_memref_types(%arg0: memref<4xf32>, %arg1: memref<4x4xf32>) {
  return
}

// TEST_2: func.func @test_memref_types([[A0:%.+]]: memref<4xf32>, [[A1:%.+]]: memref<4x4xf32>) {
