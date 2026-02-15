// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir -o - | FileCheck %s


// CHECK-LABEL:   func.func @rem_ui_32(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_4:.*]] = arith.remui %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           return %[[VAL_4]] : i32
wasmssa.func exported @rem_ui_32(%arg0: !wasmssa<local ref to i32>, %arg1: !wasmssa<local ref to i32>) -> i32 {
    %v0 = wasmssa.local_get %arg0: ref to i32
    %v1 = wasmssa.local_get %arg1: ref to i32
    %rem = wasmssa.rem_ui %v0 %v1 : i32
    wasmssa.return %rem : i32
}

// CHECK-LABEL:   func.func @rem_si_32(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<i32>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_4:.*]] = arith.remsi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           return %[[VAL_4]] : i32
wasmssa.func exported @rem_si_32(%arg0: !wasmssa<local ref to i32>, %arg1: !wasmssa<local ref to i32>) -> i32 {
    %v0 = wasmssa.local_get %arg0: ref to i32
    %v1 = wasmssa.local_get %arg1: ref to i32
    %rem = wasmssa.rem_si %v0 %v1 : i32
    wasmssa.return %rem : i32
}

// CHECK-LABEL:   func.func @rem_ui_64(
// CHECK-SAME:      %[[ARG0:.*]]: i64,
// CHECK-SAME:      %[[ARG1:.*]]: i64) -> i64 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<i64>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<i64>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_4:.*]] = arith.remui %[[VAL_2]], %[[VAL_3]] : i64
// CHECK:           return %[[VAL_4]] : i64
wasmssa.func exported @rem_ui_64(%arg0: !wasmssa<local ref to i64>, %arg1: !wasmssa<local ref to i64>) -> i64 {
    %v0 = wasmssa.local_get %arg0: ref to i64
    %v1 = wasmssa.local_get %arg1: ref to i64
    %rem = wasmssa.rem_ui %v0 %v1 : i64
    wasmssa.return %rem : i64
}

// CHECK-LABEL:   func.func @rem_si_64(
// CHECK-SAME:      %[[ARG0:.*]]: i64,
// CHECK-SAME:      %[[ARG1:.*]]: i64) -> i64 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[ARG1]], %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i64>
// CHECK:           memref.store %[[ARG0]], %[[VAL_1]][] : memref<i64>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]][] : memref<i64>
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]][] : memref<i64>
// CHECK:           %[[VAL_4:.*]] = arith.remsi %[[VAL_2]], %[[VAL_3]] : i64
// CHECK:           return %[[VAL_4]] : i64
wasmssa.func exported @rem_si_64(%arg0: !wasmssa<local ref to i64>, %arg1: !wasmssa<local ref to i64>) -> i64 {
    %v0 = wasmssa.local_get %arg0: ref to i64
    %v1 = wasmssa.local_get %arg1: ref to i64
    %rem = wasmssa.rem_si %v0 %v1 : i64
    wasmssa.return %rem : i64
}
