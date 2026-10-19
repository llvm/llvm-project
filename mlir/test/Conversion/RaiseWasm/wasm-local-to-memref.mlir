// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s

// CHECK-LABEL:   func.func @func_0() -> f32 {
wasmssa.func exported @func_0() -> f32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<f32>
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           memref.store %[[VAL_1]], %[[VAL_0]][] : memref<f32>
  %0 = wasmssa.local of type f32
// CHECK:           %[[VAL_2:.*]] = memref.alloca() : memref<f32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           memref.store %[[VAL_3]], %[[VAL_2]][] : memref<f32>
  %1 = wasmssa.local of type f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 8.000000e+00 : f32
  %2 = wasmssa.const 8.000000e+00 : f32
// CHECK:           memref.store %[[VAL_4]], %[[VAL_0]][] : memref<f32>
  wasmssa.local_set %0 : ref to f32 to %2 : f32
// CHECK:           %[[VAL_5:.*]] = memref.load %[[VAL_0]][] : memref<f32>
  %3 = wasmssa.local_get %0 : ref to f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1.200000e+01 : f32
  %4 = wasmssa.const 1.200000e+01 : f32
// CHECK:           memref.store %[[VAL_6]], %[[VAL_2]][] : memref<f32>
  %5 = wasmssa.local_tee %1 : ref to f32 to %4 : f32
// CHECK:           %[[VAL_7:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
  %6 = wasmssa.add %3 %5 : f32
// CHECK:           return %[[VAL_7]] : f32
  wasmssa.return %6 : f32
}

// CHECK-LABEL:   func.func @func_1() -> i32 {
wasmssa.func exported @func_1() -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           memref.store %[[VAL_1]], %[[VAL_0]][] : memref<i32>
  %0 = wasmssa.local of type i32
// CHECK:           %[[VAL_2:.*]] = memref.alloca() : memref<i32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           memref.store %[[VAL_3]], %[[VAL_2]][] : memref<i32>
  %1 = wasmssa.local of type i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 8 : i32
  %2 = wasmssa.const 8 : i32
// CHECK:           memref.store %[[VAL_4]], %[[VAL_0]][] : memref<i32>
  wasmssa.local_set %0 : ref to i32 to %2 : i32
// CHECK:           %[[VAL_5:.*]] = memref.load %[[VAL_0]][] : memref<i32>
  %3 = wasmssa.local_get %0 : ref to i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 12 : i32
  %4 = wasmssa.const 12 : i32
// CHECK:           memref.store %[[VAL_6]], %[[VAL_2]][] : memref<i32>
  %5 = wasmssa.local_tee %1 : ref to i32 to %4 : i32
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : i32
  %6 = wasmssa.add %3 %5 : i32
// CHECK:           return %[[VAL_7]] : i32
  wasmssa.return %6 : i32
}

// CHECK-LABEL:   func.func @func_2(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32) -> i32 {
wasmssa.func exported @func_2(%arg0: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : i32
  %1 = wasmssa.const 3 : i32
// CHECK:           memref.store %[[VAL_2]], %[[VAL_1]][] : memref<i32>
  wasmssa.local_set %arg0 : ref to i32 to %1 : i32
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_1]][] : memref<i32>
  %2 = wasmssa.local_get %arg0 : ref to i32
// CHECK:           return %[[VAL_3]] : i32
  wasmssa.return %2 : i32
}
