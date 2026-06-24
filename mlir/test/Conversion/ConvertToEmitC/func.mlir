// RUN: mlir-opt -convert-to-emitc %s | FileCheck %s

// CHECK-LABEL emitc.func @int(%[[ARG:.*]]: i32) -> i32
func.func @int(%arg0: i32) -> i32 {
    // CHECK: return
    return %arg0 : i32
}

// CHECK-LABEL emitc.func @index(%[[ARG:.*]]: !emitc.size_t) -> !emitc.size_t
func.func @index(%arg0: index) -> index {
    // CHECK: return
    return %arg0 : index
}

// CHECK-LABEL: emitc.func @call_with_one_result
func.func @call_with_one_result(%arg0: index) -> index {
    // CHECK: call @index
    // CHECK-SAME: (!emitc.size_t) -> !emitc.size_t
    %0 = func.call @index(%arg0) : (index) -> index
    return %0 : index
}
