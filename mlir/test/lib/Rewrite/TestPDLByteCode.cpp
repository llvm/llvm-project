//===- TestPDLByteCode.cpp - Test rewriter bytecode functionality ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

/// Custom constraint invoked from PDL.
static LogicalResult customSingleEntityConstraint(PatternRewriter &rewriter,
                                                  Operation *rootOp) {
  return success(rootOp->getName().getStringRef() == "test.op");
}
static LogicalResult customMultiEntityConstraint(PatternRewriter &rewriter,
                                                 Operation *root,
                                                 Operation *rootCopy) {
  return customSingleEntityConstraint(rewriter, rootCopy);
}
static LogicalResult customMultiEntityVariadicConstraint(
    PatternRewriter &rewriter, ValueRange operandValues, TypeRange typeValues) {
  if (operandValues.size() != 2 || typeValues.size() != 2)
    return failure();
  return success();
}

// Custom constraint that returns a value if the op is named test.success_op
static LogicalResult customValueResultConstraint(PatternRewriter &rewriter,
                                                 PDLResultList &results,
                                                 ArrayRef<PDLValue> args) {
  auto *op = args[0].cast<Operation *>();
  if (op->getName().getStringRef() == "test.success_op") {
    StringAttr customAttr = rewriter.getStringAttr("test.success");
    results.push_back(customAttr);
    return success();
  }
  return failure();
}

// Custom constraint that returns a type if the op is named test.success_op
static LogicalResult customTypeResultConstraint(PatternRewriter &rewriter,
                                                PDLResultList &results,
                                                ArrayRef<PDLValue> args) {
  auto *op = args[0].cast<Operation *>();
  if (op->getName().getStringRef() == "test.success_op") {
    results.push_back(rewriter.getF32Type());
    return success();
  }
  return failure();
}

// Custom constraint that returns a type range of variable length if the op is
// named test.success_op
static LogicalResult customTypeRangeResultConstraint(PatternRewriter &rewriter,
                                                     PDLResultList &results,
                                                     ArrayRef<PDLValue> args) {
  auto *op = args[0].cast<Operation *>();
  int numTypes = cast<IntegerAttr>(args[1].cast<Attribute>()).getInt();

  if (op->getName().getStringRef() == "test.success_op") {
    SmallVector<Type> types;
    for (int i = 0; i < numTypes; i++) {
      types.push_back(rewriter.getF32Type());
    }
    results.push_back(TypeRange(types));
    return success();
  }
  return failure();
}

// Custom creator invoked from PDL.
static Operation *customCreate(PatternRewriter &rewriter, Operation *op) {
  return rewriter.create(OperationState(op->getLoc(), "test.success"));
}
static auto customVariadicResultCreate(PatternRewriter &rewriter,
                                       Operation *root) {
  return std::make_pair(root->getOperands(), root->getOperands().getTypes());
}
static Type customCreateType(PatternRewriter &rewriter) {
  return rewriter.getF32Type();
}
static std::string customCreateStrAttr(PatternRewriter &rewriter) {
  return "test.str";
}

/// Custom rewriter invoked from PDL.
static void customRewriter(PatternRewriter &rewriter, Operation *root,
                           Value input) {
  rewriter.create(root->getLoc(), rewriter.getStringAttr("test.success"),
                  input);
  rewriter.eraseOp(root);
}

namespace {
struct TestPDLByteCodePass
    : public PassWrapper<TestPDLByteCodePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPDLByteCodePass)

  StringRef getArgument() const final { return "test-pdl-bytecode-pass"; }
  StringRef getDescription() const final {
    return "Test PDL ByteCode functionality";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    // Mark the pdl_interp dialect as a dependent. This is needed, because we
    // create ops from that dialect as a part of the PDL-to-PDLInterp lowering.
    registry.insert<pdl_interp::PDLInterpDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();

    // The test cases are encompassed via two modules, one containing the
    // patterns and one containing the operations to rewrite.
    ModuleOp patternModule = module.lookupSymbol<ModuleOp>(
        StringAttr::get(module->getContext(), "patterns"));
    ModuleOp irModule = module.lookupSymbol<ModuleOp>(
        StringAttr::get(module->getContext(), "ir"));
    if (!patternModule || !irModule)
      return;

    RewritePatternSet patternList(module->getContext());

    // Register ahead of time to test when functions are registered without a
    // pattern.
    patternList.getPDLPatterns().registerConstraintFunction(
        "multi_entity_constraint", customMultiEntityConstraint);
    patternList.getPDLPatterns().registerConstraintFunction(
        "single_entity_constraint", customSingleEntityConstraint);

    // Process the pattern module.
    patternModule.getOperation()->remove();
    PDLPatternModule pdlPattern(patternModule);

    // Note: This constraint was already registered, but we re-register here to
    // ensure that duplication registration is allowed (the duplicate mapping
    // will be ignored). This tests that we support separating the registration
    // of library functions from the construction of patterns, and also that we
    // allow multiple patterns to depend on the same library functions (without
    // asserting/crashing).
    pdlPattern.registerConstraintFunction("multi_entity_constraint",
                                          customMultiEntityConstraint);
    pdlPattern.registerConstraintFunction("multi_entity_var_constraint",
                                          customMultiEntityVariadicConstraint);
    pdlPattern.registerConstraintFunction("op_constr_return_attr",
                                          customValueResultConstraint);
    pdlPattern.registerConstraintFunction("op_constr_return_type",
                                          customTypeResultConstraint);
    pdlPattern.registerConstraintFunction("op_constr_return_type_range",
                                          customTypeRangeResultConstraint);
    pdlPattern.registerRewriteFunction("creator", customCreate);
    pdlPattern.registerRewriteFunction("var_creator",
                                       customVariadicResultCreate);
    pdlPattern.registerRewriteFunction("type_creator", customCreateType);
    pdlPattern.registerRewriteFunction("str_creator", customCreateStrAttr);
    pdlPattern.registerRewriteFunction("rewriter", customRewriter);
    patternList.add(std::move(pdlPattern));

    // Invoke the pattern driver with the provided patterns.
    (void)applyPatternsGreedily(irModule.getBodyRegion(),
                                std::move(patternList));
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestPDLByteCodePass() { PassRegistration<TestPDLByteCodePass>(); }
} // namespace test
} // namespace mlir
