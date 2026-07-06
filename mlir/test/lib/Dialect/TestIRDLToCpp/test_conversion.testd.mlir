// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-irdl-conversion-check)" | FileCheck %s
// CHECK-LABEL: module {
module {
    // CHECK: func.func @test(%[[test_arg:[^ ]*]]: i1) {
    // CHECK: %[[v0:[^ ]*]] = "test_irdl_to_cpp.bar"() : () -> i32
    // CHECK: %[[v1:[^ ]*]] = "test_irdl_to_cpp.bar"() : () -> i32
    // CHECK: %[[v2:[^ ]*]] = "test_irdl_to_cpp.hash"(%[[v0]], %[[v0]]) : (i32, i32) -> i32
    // CHECK: scf.if %[[test_arg]]
    // CHECK: return
    // CHECK: }
    func.func @test(%test_arg: i1) {
        %0 = "test_irdl_to_cpp.bar"() : () -> i32
        %1 = "test_irdl_to_cpp.beef"(%0, %0) : (i32, i32) -> i32
        "test_irdl_to_cpp.conditional"(%test_arg) ({
        ^cond(%test: i1):
          %3 = "test_irdl_to_cpp.bar"() : () -> i32
          "test.terminator"() : ()->()
        }, {
        ^then(%what: i1, %ever: i32):
          %4 = "test_irdl_to_cpp.bar"() : () -> i32
          "test.terminator"() : ()->()
        }, {
        ^else():
          %5 = "test_irdl_to_cpp.bar"() : () -> i32
          "test.terminator"() : ()->()
        }) : (i1) -> ()
        return
    }

// CHECK: }
}
