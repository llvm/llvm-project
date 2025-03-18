// RUN: mlir-irdl-to-cpp %s
// CHECK-LABEL: irdl.dialect @test_irdl_to_cpp {
irdl.dialect @test_irdl_to_cpp {
    // CHECK: irdl.type @foo
    irdl.type @foo

    // CHECK: irdl.type @bar {
    // CHECK: %[[v0:[^ ]*]] = irdl.any
    // CHECK: irdl.results(res: %[[v0]])
    // CHECK: }
    irdl.operation @bar {
        %0 = irdl.any
        irdl.results(res: %0)
    }


    // CHECK: irdl.type @beef {
    // CHECK: %[[v0:[^ ]*]] = irdl.any
    // CHECK: irdl.operands(lhs: %[[v0]], rhs: %[[v0]])
    // CHECK: irdl.results(res: %[[v0]])
    // CHECK: }
    irdl.operation @beef {
        %0 = irdl.any
        irdl.operands(lhs: %0, rhs: %0)
        irdl.results(res: %0)
    }

    // CHECK: irdl.type @hash {
    // CHECK: %[[v0:[^ ]*]] = irdl.any
    // CHECK: irdl.operands(lhs: %[[v0]], rhs: %[[v0]])
    // CHECK: irdl.results(res: %[[v0]])
    // CHECK: }
    irdl.operation @hash {
        %0 = irdl.any
        irdl.operands(lhs: %0, rhs: %0)
        irdl.results(res: %0)
    }
}
