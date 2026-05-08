// RUN: mlir-opt %s | FileCheck %s

module {
  wasmssa.func @select_i32(%cond: !wasmssa<local ref to i32>,
                           %a: !wasmssa<local ref to i32>,
                           %b: !wasmssa<local ref to i32>) -> i32 {
    %0 = wasmssa.local_get %cond : ref to i32
    %1 = wasmssa.local_get %a : ref to i32
    %2 = wasmssa.local_get %b : ref to i32
    %r = wasmssa.select %0, %1, %2 : i32
    wasmssa.return %r : i32
  }

  wasmssa.func @select_f64(%cond: !wasmssa<local ref to i32>,
                           %a: !wasmssa<local ref to f64>,
                           %b: !wasmssa<local ref to f64>) -> f64 {
    %0 = wasmssa.local_get %cond : ref to i32
    %1 = wasmssa.local_get %a : ref to f64
    %2 = wasmssa.local_get %b : ref to f64
    %r = wasmssa.select %0, %1, %2 : f64
    wasmssa.return %r : f64
  }
}

// CHECK-LABEL:   wasmssa.func @select_i32(
// CHECK:           %[[COND:.*]] = wasmssa.local_get
// CHECK:           %[[A:.*]] = wasmssa.local_get
// CHECK:           %[[B:.*]] = wasmssa.local_get
// CHECK:           %[[R:.*]] = wasmssa.select %[[COND]], %[[A]], %[[B]] : i32
// CHECK:           wasmssa.return %[[R]] : i32

// CHECK-LABEL:   wasmssa.func @select_f64(
// CHECK:           %[[COND:.*]] = wasmssa.local_get
// CHECK:           %[[A:.*]] = wasmssa.local_get
// CHECK:           %[[B:.*]] = wasmssa.local_get
// CHECK:           %[[R:.*]] = wasmssa.select %[[COND]], %[[A]], %[[B]] : f64
// CHECK:           wasmssa.return %[[R]] : f64
