// RUN: mlir-opt -convert-to-emitc %s | FileCheck %s

// CHECK-LABEL emitc.func @int(%[[ARG:.*]]: i32)
func.func @int(%arg0: i32) {
    // CHECK: return
    return
}

// CHECK-LABEL emitc.func @index(%[[ARG:.*]]: !emitc.size_t)
func.func @index(%arg0: index) {
    // CHECK: return
    return
}

// CHECK-LABEL emitc.func @memref(%[[ARG:.*]]: !emitc.array<1xf32>)
func.func @memref(%arg0: memref<1xf32>) {
    // CHECK: return
    return
}
