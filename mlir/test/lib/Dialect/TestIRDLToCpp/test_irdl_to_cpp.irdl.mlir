// RUN: mlir-irdl-to-cpp %s | FileCheck %s

// CHECK: class TestIrdlToCpp
irdl.dialect @test_irdl_to_cpp {
    
    // CHECK: class FooType
    irdl.type @foo

    // CHECK: class BarOp
    // CHECK: ::mlir::Value getRes()
    irdl.operation @bar {
        %0 = irdl.any
        irdl.results(res: %0)
    }

    // CHECK: class BeefOp
    // CHECK: ::mlir::Value getLhs()
    // CHECK: ::mlir::Value getRhs()
    // CHECK: ::mlir::Value getRes()
    irdl.operation @beef {
        %0 = irdl.any
        irdl.operands(lhs: %0, rhs: %0)
        irdl.results(res: %0)
    }

    // CHECK: class HashOp
    // CHECK: ::mlir::Value getLhs()
    // CHECK: ::mlir::Value getRhs()
    // CHECK: ::mlir::Value getRes()
    irdl.operation @hash {
        %0 = irdl.any
        irdl.operands(lhs: %0, rhs: %0)
        irdl.results(res: %0)
    }
}
