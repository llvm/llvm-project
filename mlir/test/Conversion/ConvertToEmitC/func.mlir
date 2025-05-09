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
