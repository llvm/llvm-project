// RUN: mlir-opt %s --raise-wasm-mlir | FileCheck %s


// CHECK-LABEL:   func.func @callee(
// CHECK-SAME:                      %[[ARG0:.*]]: i32) -> i32 {
wasmssa.func exported @callee(%arg0: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<i32>
%v0 = wasmssa.local_get %arg0 : ref to i32
// CHECK:           return %[[VAL_1]] : i32
wasmssa.return %v0 : i32
}

wasmssa.func exported @caller(%arg0: !wasmssa<local ref to i32>) -> i32 {
// CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[ARG0]], %[[VAL_0]][] : memref<i32>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]][] : memref<i32>
%v0 = wasmssa.local_get %arg0 : ref to i32
// CHECK:           %[[VAL_2:.*]] = call @callee(%[[VAL_1]]) : (i32) -> i32
%0 = wasmssa.call @callee (%v0) : (i32) -> i32
// CHECK:           return %[[VAL_2]] : i32
wasmssa.return %0 : i32
}

// CHECK-LABEL:         func.func private @"my_module::foo"() -> i32
wasmssa.import_func "foo" from "my_module" as @func_0 {sym_visibility = "nested", type = () -> (i32)}

// CHECK-LABEL:   func.func @user_of_func0() -> i32 {
wasmssa.func exported @user_of_func0() -> i32 {
// CHECK:           %[[VAL_0:.*]] = call @"my_module::foo"() : () -> i32
%0 = wasmssa.call @func_0 : () -> i32
// CHECK:           return %[[VAL_0]] : i32
wasmssa.return %0 : i32
}
