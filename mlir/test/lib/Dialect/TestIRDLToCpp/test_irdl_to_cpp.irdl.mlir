// RUN: mlir-irdl-to-cpp %s | FileCheck %s

// CHECK: namespace mlir {
// CHECK: namespace test_irdl_to_cpp {

// CHECK: class TestIrdlToCpp
// CHECK: } // namespace test_irdl_to_cpp
// CHECK: } // namespace mlir
irdl.dialect @test_irdl_to_cpp {

    // CHECK: class FooType
    irdl.type @foo

    // CHECK: class _8Type
    irdl.type @"8"

    // CHECK: class BarOp;
    // CHECK-NEXT: namespace nested {
    // CHECK-NEXT: class NamespacedOp;
    // CHECK-NEXT: }
    // CHECK-NEXT: namespace nested::namespaced {
    // CHECK-NEXT: class MoreOp;
    // CHECK-NEXT: }
    // CHECK-NEXT: class BeefOp;

    // CHECK: namespace wmma::f16::_16x16x64 {
    // CHECK-NEXT: class Bf8BfOp;
    // CHECK-NEXT: }
    // CHECK-NEXT: namespace _7::_8 {
    // CHECK-NEXT: class _9Op;
    // CHECK-NEXT: }
    // CHECK-NEXT: class _10Op;

    // CHECK: class BarOp
    // CHECK: ::mlir::Value getRes()
    irdl.operation @bar {
        %0 = irdl.any
        irdl.results(res: %0)
    }

    // CHECK: // ::mlir::test_irdl_to_cpp::nested::NamespacedOp declarations
    // CHECK: namespace nested {
    // CHECK: class NamespacedOp : public ::mlir::Op<NamespacedOp
    // CHECK: static constexpr ::llvm::StringLiteral getOperationName()
    // CHECK-NEXT: return ::llvm::StringLiteral("test_irdl_to_cpp.nested.namespaced");
    // CHECK: } // namespace nested
    // CHECK: MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::test_irdl_to_cpp::nested::NamespacedOp)
    irdl.operation @nested.namespaced {
        %0 = irdl.any
        irdl.results(res: %0)
    }

    // CHECK: // ::mlir::test_irdl_to_cpp::nested::namespaced::MoreOp declarations
    // CHECK: namespace nested {
    // CHECK: namespace namespaced {
    // CHECK: class MoreOp : public ::mlir::Op<MoreOp
    // CHECK: static constexpr ::llvm::StringLiteral getOperationName()
    // CHECK-NEXT: return ::llvm::StringLiteral("test_irdl_to_cpp.nested.namespaced.more");
    // CHECK: } // namespace namespaced
    // CHECK: } // namespace nested
    // CHECK: MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::test_irdl_to_cpp::nested::namespaced::MoreOp)
    irdl.operation @nested.namespaced.more {
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

    // CHECK: // ::mlir::test_irdl_to_cpp::wmma::f16::_16x16x64::Bf8BfOp declarations
    // CHECK: namespace wmma {
    // CHECK-NEXT: namespace f16 {
    // CHECK-NEXT: namespace _16x16x64 {
    // CHECK: class Bf8BfOp : public ::mlir::Op<Bf8BfOp
    // CHECK: return ::llvm::StringLiteral("test_irdl_to_cpp.wmma.f16.16x16x64.bf8_bf");
    // CHECK: } // namespace _16x16x64
    // CHECK: MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::test_irdl_to_cpp::wmma::f16::_16x16x64::Bf8BfOp)
    irdl.operation @wmma.f16.16x16x64.bf8_bf

    // CHECK: // ::mlir::test_irdl_to_cpp::_7::_8::_9Op declarations
    // CHECK: namespace _7 {
    // CHECK-NEXT: namespace _8 {
    // CHECK: class _9Op : public ::mlir::Op<_9Op
    // CHECK: return ::llvm::StringLiteral("test_irdl_to_cpp.7.8.9");
    // CHECK: MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::test_irdl_to_cpp::_7::_8::_9Op)
    irdl.operation @"7.8.9"

    // CHECK: class _10Op : public ::mlir::Op<_10Op
    // CHECK: return ::llvm::StringLiteral("test_irdl_to_cpp.10");
    irdl.operation @"10"

    // CHECK: // ::mlir::test_irdl_to_cpp::nested::NamespacedOp definitions
    // CHECK: namespace nested {
    // CHECK: NamespacedOp::build
    // CHECK: } // namespace nested
    // CHECK: MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::test_irdl_to_cpp::nested::NamespacedOp)

    // CHECK: // ::mlir::test_irdl_to_cpp::nested::namespaced::MoreOp definitions
    // CHECK: namespace nested {
    // CHECK: namespace namespaced {
    // CHECK: MoreOp::build
    // CHECK: } // namespace namespaced
    // CHECK: } // namespace nested
    // CHECK: MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::test_irdl_to_cpp::nested::namespaced::MoreOp)

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

    // CHECK: // ::mlir::test_irdl_to_cpp::wmma::f16::_16x16x64::Bf8BfOp definitions
    // CHECK: namespace _16x16x64 {
    // CHECK: Bf8BfOp::build
    // CHECK: } // namespace _16x16x64
    // CHECK: MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::test_irdl_to_cpp::wmma::f16::_16x16x64::Bf8BfOp)
    // CHECK: _9Op::build
    // CHECK: MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::test_irdl_to_cpp::_7::_8::_9Op)
    // CHECK: _10Op::build
    // CHECK: void TestIrdlToCppDialect::initialize()
    // CHECK: ::mlir::test_irdl_to_cpp::nested::NamespacedOp
    // CHECK: ::mlir::test_irdl_to_cpp::nested::namespaced::MoreOp
    // CHECK: ::mlir::test_irdl_to_cpp::wmma::f16::_16x16x64::Bf8BfOp,
    // CHECK-NEXT: ::mlir::test_irdl_to_cpp::_7::_8::_9Op,
    // CHECK-NEXT: ::mlir::test_irdl_to_cpp::_10Op
}
