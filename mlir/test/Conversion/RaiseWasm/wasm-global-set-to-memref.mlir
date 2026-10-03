// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s

wasmssa.global @global_i32 i32 mutable : {
  %0 = wasmssa.const 10 : i32
  wasmssa.return %0 : i32
}

wasmssa.global @global_f64 f64 mutable : {
  %0 = wasmssa.const 3.14 : f64
  wasmssa.return %0 : f64
}

// CHECK-LABEL:   func.func @set_global_i32() {
wasmssa.func exported @set_global_i32() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 42 : i32
  %0 = wasmssa.const 42 : i32
// CHECK:           %[[VAL_1:.*]] = memref.get_global @global_i32 : memref<1xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]]{{\[}}%[[VAL_2]]] : memref<1xi32>
  wasmssa.global_set @global_i32 to %0 : i32
// CHECK:           return
  wasmssa.return
}

// CHECK-LABEL:   func.func @set_global_f64() {
wasmssa.func exported @set_global_f64() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2.500000e-01 : f64
  %0 = wasmssa.const 0.25 : f64
// CHECK:           %[[VAL_1:.*]] = memref.get_global @global_f64 : memref<1xf64>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]]{{\[}}%[[VAL_2]]] : memref<1xf64>
  wasmssa.global_set @global_f64 to %0 : f64
// CHECK:           return
  wasmssa.return
}
