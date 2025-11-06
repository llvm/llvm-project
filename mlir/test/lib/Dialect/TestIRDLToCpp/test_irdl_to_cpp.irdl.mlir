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

    // CHECK: ConditionalOp declarations
    // CHECK: ConditionalOpGenericAdaptorBase
    // CHECK:  ::mlir::Region &getCond() { return *getRegions()[0]; }
    // CHECK:  ::mlir::Region &getThen() { return *getRegions()[1]; }
    // CHECK:  ::mlir::Region &getElse() { return *getRegions()[2]; }
    //
    // CHECK: class ConditionalOp : public ::mlir::Op<ConditionalOp, ::mlir::OpTrait::NRegions<3>::Impl, ::mlir::OpTrait::OpInvariants>
    // CHECK:  ::mlir::Region &getCond() { return (*this)->getRegion(0); }
    // CHECK:  ::mlir::Region &getThen() { return (*this)->getRegion(1); }
    // CHECK:  ::mlir::Region &getElse() { return (*this)->getRegion(2); }

    // CHECK: ConditionalOp definitions
    // CHECK: __mlir_irdl_local_region_constraint_ConditionalOp_cond
    // CHECK: if (!(region.getNumArguments() == 1)) {
    // CHECK: failed to verify constraint: region with 1 entry block argument(s)

    // CHECK: __mlir_irdl_local_region_constraint_ConditionalOp_then
    // CHECK: if (!(true)) {

    // CHECK: __mlir_irdl_local_region_constraint_ConditionalOp_else
    // CHECK: if (!(region.getNumArguments() == 0)) {
    // CHECK: failed to verify constraint: region with 0 entry block argument(s)

    // CHECK:  ConditionalOp::build
    // CHECK: for (unsigned i = 0; i != 3; ++i)
    // CHECK-NEXT: (void)odsState.addRegion();

    // CHECK: ConditionalOp::verifyInvariantsImpl
    // CHECK: __mlir_irdl_local_region_constraint_ConditionalOp_cond
    // CHECK: failure
    // CHECK: __mlir_irdl_local_region_constraint_ConditionalOp_then
    // CHECK: failure
    // CHECK: __mlir_irdl_local_region_constraint_ConditionalOp_else
    // CHECK: failure
    // CHECK: success
    irdl.operation @conditional {
        %r0 = irdl.region      // Unconstrained region
        %r1 = irdl.region()    // Region with no entry block arguments

        // TODO(#161018): support irdl.is in irdl-to-cpp
        // %v0 = irdl.is i1       // Type constraint: i1 (boolean)
        %v0 = irdl.any
        %r2 = irdl.region(%v0) // Region with one i1 entry block argument
        irdl.regions(cond: %r2, then: %r0, else: %r1)

        %0 = irdl.any
        irdl.operands(input: %0)
    }
}
