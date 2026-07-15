// RUN: mlir-opt %s | FileCheck %s

module {
  wasmssa.func @func_0() -> f32 {
    %0 = wasmssa.local of type f32
    %1 = wasmssa.local of type f32
    %2 = wasmssa.const 8.000000e+00 : f32
    %3 = wasmssa.const 1.200000e+01 : f32
    %4 = wasmssa.add %2 %3 : f32
    wasmssa.return %4 : f32
  }
  wasmssa.func @func_1() -> i32 {
    %0 = wasmssa.local of type i32
    %1 = wasmssa.local of type i32
    %2 = wasmssa.const 8 : i32
    %3 = wasmssa.const 12 : i32
    %4 = wasmssa.add %2 %3 : i32
    wasmssa.return %4 : i32
  }
  wasmssa.func @func_2(%arg0: !wasmssa<local ref to i32>) -> i32 {
    %0 = wasmssa.const 3 : i32
    wasmssa.return %0 : i32
  }
}

// CHECK-LABEL:   wasmssa.func @func_0() -> f32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local of type f32
// CHECK:           %[[VAL_1:.*]] = wasmssa.local of type f32
// CHECK:           %[[VAL_2:.*]] = wasmssa.const 8.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = wasmssa.const 1.200000e+01 : f32
// CHECK:           %[[VAL_4:.*]] = wasmssa.add %[[VAL_2]] %[[VAL_3]] : f32
// CHECK:           wasmssa.return %[[VAL_4]] : f32

// CHECK-LABEL:   wasmssa.func @func_1() -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local of type i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.local of type i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.const 8 : i32
// CHECK:           %[[VAL_3:.*]] = wasmssa.const 12 : i32
// CHECK:           %[[VAL_4:.*]] = wasmssa.add %[[VAL_2]] %[[VAL_3]] : i32
// CHECK:           wasmssa.return %[[VAL_4]] : i32

// CHECK-LABEL:   wasmssa.func @func_2(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.const 3 : i32
// CHECK:           wasmssa.return %[[VAL_0]] : i32
