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
static LogicalResult customSingleEntityConstraint(PDLValue value,
                                                  ArrayAttr constantParams,
                                                  PatternRewriter &rewriter) {
  Operation *rootOp = value.cast<Operation *>();
  return success(rootOp->getName().getStringRef() == "test.op");
}
static LogicalResult customMultiEntityConstraint(ArrayRef<PDLValue> values,
                                                 ArrayAttr constantParams,
                                                 PatternRewriter &rewriter) {
  return customSingleEntityConstraint(values[1], constantParams, rewriter);
}
static LogicalResult
customMultiEntityVariadicConstraint(ArrayRef<PDLValue> values,
                                    ArrayAttr constantParams,
                                    PatternRewriter &rewriter) {
  if (llvm::any_of(values, [](const PDLValue &value) { return !value; }))
    return failure();
  ValueRange operandValues = values[0].cast<ValueRange>();
  TypeRange typeValues = values[1].cast<TypeRange>();
  if (operandValues.size() != 2 || typeValues.size() != 2)
    return failure();
  return success();
}

// Custom creator invoked from PDL.
static void customCreate(ArrayRef<PDLValue> args, ArrayAttr constantParams,
                         PatternRewriter &rewriter, PDLResultList &results) {
  results.push_back(rewriter.createOperation(
      OperationState(args[0].cast<Operation *>()->getLoc(), "test.success")));
}
static void customVariadicResultCreate(ArrayRef<PDLValue> args,
                                       ArrayAttr constantParams,
                                       PatternRewriter &rewriter,
                                       PDLResultList &results) {
  Operation *root = args[0].cast<Operation *>();
  results.push_back(root->getOperands());
  results.push_back(root->getOperands().getTypes());
}
static void customCreateType(ArrayRef<PDLValue> args, ArrayAttr constantParams,
                             PatternRewriter &rewriter,
                             PDLResultList &results) {
  results.push_back(rewriter.getF32Type());
}

/// Custom rewriter invoked from PDL.
static void customRewriter(ArrayRef<PDLValue> args, ArrayAttr constantParams,
                           PatternRewriter &rewriter, PDLResultList &results) {
  Operation *root = args[0].cast<Operation *>();
  OperationState successOpState(root->getLoc(), "test.success");
  successOpState.addOperands(args[1].cast<Value>());
  successOpState.addAttribute("constantParams", constantParams);
  rewriter.createOperation(successOpState);
  rewriter.eraseOp(root);
}

namespace {
struct TestPDLByteCodePass
    : public PassWrapper<TestPDLByteCodePass, OperationPass<ModuleOp>> {
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
    pdlPattern.registerRewriteFunction("creator", customCreate);
    pdlPattern.registerRewriteFunction("var_creator",
                                       customVariadicResultCreate);
    pdlPattern.registerRewriteFunction("type_creator", customCreateType);
    pdlPattern.registerRewriteFunction("rewriter", customRewriter);
    patternList.add(std::move(pdlPattern));

    // Invoke the pattern driver with the provided patterns.
    (void)applyPatternsAndFoldGreedily(irModule.getBodyRegion(),
                                       std::move(patternList));
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestPDLByteCodePass() { PassRegistration<TestPDLByteCodePass>(); }
} // namespace test
} // namespace mlir
