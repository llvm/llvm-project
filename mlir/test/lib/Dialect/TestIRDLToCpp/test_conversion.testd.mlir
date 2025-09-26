// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-irdl-conversion-check)" | FileCheck %s
// CHECK-LABEL: module {
module {
    // CHECK: func.func @test() {
    // CHECK: %[[v0:[^ ]*]] = "test_irdl_to_cpp.bar"() : () -> i32
    // CHECK: %[[v1:[^ ]*]] = "test_irdl_to_cpp.bar"() : () -> i32
    // CHECK: %[[v2:[^ ]*]] = "test_irdl_to_cpp.hash"(%[[v0]], %[[v0]]) : (i32, i32) -> i32
    // CHECK: return
    // CHECK: }
    func.func @test() {
        %0 = "test_irdl_to_cpp.bar"() : () -> i32
        %1 = "test_irdl_to_cpp.beef"(%0, %0) : (i32, i32) -> i32
        return
    }

// CHECK: }
}
