//===- TestPDLByteCode.cpp - Test rewriter bytecode functionality ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

// Custom creator invoked from PDL.
static void customCreate(ArrayRef<PDLValue> args, ArrayAttr constantParams,
                         PatternRewriter &rewriter, PDLResultList &results) {
  results.push_back(rewriter.createOperation(
      OperationState(args[0].cast<Operation *>()->getLoc(), "test.success")));
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
  void runOnOperation() final {
    ModuleOp module = getOperation();

    // The test cases are encompassed via two modules, one containing the
    // patterns and one containing the operations to rewrite.
    ModuleOp patternModule = module.lookupSymbol<ModuleOp>("patterns");
    ModuleOp irModule = module.lookupSymbol<ModuleOp>("ir");
    if (!patternModule || !irModule)
      return;

    // Process the pattern module.
    patternModule.getOperation()->remove();
    PDLPatternModule pdlPattern(patternModule);
    pdlPattern.registerConstraintFunction("multi_entity_constraint",
                                          customMultiEntityConstraint);
    pdlPattern.registerConstraintFunction("single_entity_constraint",
                                          customSingleEntityConstraint);
    pdlPattern.registerRewriteFunction("creator", customCreate);
    pdlPattern.registerRewriteFunction("rewriter", customRewriter);

    OwningRewritePatternList patternList(std::move(pdlPattern));

    // Invoke the pattern driver with the provided patterns.
    (void)applyPatternsAndFoldGreedily(irModule.getBodyRegion(),
                                       std::move(patternList));
  }
};
} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestPDLByteCodePass() {
  PassRegistration<TestPDLByteCodePass>("test-pdl-bytecode-pass",
                                        "Test PDL ByteCode functionality");
}
} // namespace test
} // namespace mlir
