// RUN: mlir-opt --split-input-file %s --raise-wasm-mlir | FileCheck %s

wasmssa.global @global_i32 i32 mutable : {
  %0 = wasmssa.const 66560 : i32
  wasmssa.return %0 : i32
}

wasmssa.global @global_i64 i64 mutable : {
  %0 = wasmssa.const 37017 : i64
  wasmssa.return %0 : i64
}

wasmssa.global @global_f32 f32 mutable : {
  %0 = wasmssa.const 0.125 : f32
  wasmssa.return %0 : f32
}

wasmssa.global @global_f64 f64 mutable : {
  %0 = wasmssa.const 3.14 : f64
  wasmssa.return %0 : f64
}

wasmssa.global @global_user0 i32 mutable : {
  %0 = wasmssa.global_get @global_top_define : i32
  wasmssa.return %0 : i32
}

wasmssa.import_global "extern_global_var" from "module" as @global_top_define nested : i32

// CHECK-LABEL:   memref.global "private" @global_i32 : memref<1xi32> = dense<66560>
// CHECK:         memref.global "private" @global_i64 : memref<1xi64> = dense<37017>
// CHECK:         memref.global "private" @global_f32 : memref<1xf32> = dense<1.250000e-01>
// CHECK:         memref.global "private" @global_f64 : memref<1xf64> = dense<3.140000e+00>
// CHECK:         memref.global "private" @global_user0 : memref<1xi32> = uninitialized

// CHECK-LABEL:   func.func @"global_user0::initializer"() attributes {initializer} {
// CHECK:           %[[VAL_0:.*]] = memref.get_global @"module::extern_global_var" : memref<1xi32>
// CHECK:           %[[VAL_1:.*]] = memref.get_global @global_user0 : memref<1xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_2]]] : memref<1xi32>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_1]]{{\[}}%[[VAL_2]]] : memref<1xi32>
// CHECK:           return
// CHECK:         memref.global "nested" constant @"module::extern_global_var" : memref<1xi32>
