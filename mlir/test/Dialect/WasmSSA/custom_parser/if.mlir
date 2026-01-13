// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL:   wasmssa.func @func_0(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:           wasmssa.if %[[VAL_0]] : {
// CHECK:             %[[VAL_1:.*]] = wasmssa.const 5.000000e-01 : f32
// CHECK:             wasmssa.block_return %[[VAL_1]] : f32
// CHECK:           } "else "{
// CHECK:             %[[VAL_2:.*]] = wasmssa.const 2.500000e-01 : f32
// CHECK:             wasmssa.block_return %[[VAL_2]] : f32
// CHECK:           }> ^bb1
// CHECK:         ^bb1(%[[VAL_3:.*]]: f32):
// CHECK:           wasmssa.return %[[VAL_3]] : f32
wasmssa.func @func_0(%arg0 : !wasmssa<local ref to i32>) -> i32 {
  %cond = wasmssa.local_get %arg0 : ref to i32
  wasmssa.if %cond : {
    %c0 = wasmssa.const 0.5 : f32
    wasmssa.block_return %c0 : f32
  } else {
   %c1 = wasmssa.const 0.25 : f32
   wasmssa.block_return %c1 : f32
  } >^bb1
 ^bb1(%retVal: f32):
  wasmssa.return %retVal : f32
}

// CHECK-LABEL:   wasmssa.func @func_1(
// CHECK-SAME:      %[[ARG0:.*]]: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = wasmssa.local_get %[[ARG0]] :  ref to i32
// CHECK:           %[[VAL_1:.*]] = wasmssa.local of type i32
// CHECK:           %[[VAL_2:.*]] = wasmssa.const 0 : i64
// CHECK:           wasmssa.if %[[VAL_0]] : {
// CHECK:             %[[VAL_3:.*]] = wasmssa.const 1 : i32
// CHECK:             wasmssa.local_set %[[VAL_1]] :  ref to i32 to %[[VAL_3]] : i32
// CHECK:             wasmssa.block_return
// CHECK:           } > ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_4:.*]] = wasmssa.local_get %[[VAL_1]] :  ref to i32
// CHECK:           wasmssa.return %[[VAL_4]] : i32
wasmssa.func @func_1(%arg0 : !wasmssa<local ref to i32>) -> i32 {
  %cond = wasmssa.local_get %arg0 : ref to i32
  %var = wasmssa.local of type i32
  %zero = wasmssa.const 0
  wasmssa.if %cond : {
    %c1 = wasmssa.const 1 : i32
    wasmssa.local_set %var : ref to i32 to %c1 : i32
    wasmssa.block_return
  } >^bb1
 ^bb1:
  %res = wasmssa.local_get %var : ref to i32
  wasmssa.return %res : i32
}
