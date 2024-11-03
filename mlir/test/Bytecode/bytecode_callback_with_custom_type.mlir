// RUN: mlir-opt %s -split-input-file --test-bytecode-callback="callback-test=1" | FileCheck %s --check-prefix=TEST_1
// RUN: mlir-opt %s -split-input-file --test-bytecode-callback="callback-test=2" | FileCheck %s --check-prefix=TEST_2

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
