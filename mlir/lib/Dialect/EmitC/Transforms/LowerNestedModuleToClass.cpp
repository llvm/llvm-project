//===- LowerNestedModuleToClass.cpp - Lower nested modules to classes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace emitc;

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_LOWERNESTEDMODULETOCLASSPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

namespace {
struct LowerNestedModuleToClassPass
    : public impl::LowerNestedModuleToClassPassBase<LowerNestedModuleToClassPass> {
  using LowerNestedModuleToClassPassBase::LowerNestedModuleToClassPassBase;
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    populateLowerNestedModuleToClassPatterns(patterns, lowerAll, useHeuristic);

    walkAndApplyPatterns(moduleOp, std::move(patterns));
  }
};
} // namespace
} // namespace emitc
} // namespace mlir

class LowerNestedModuleToClass : public OpRewritePattern<ModuleOp> {
public:
  LowerNestedModuleToClass(MLIRContext *context, bool lowerAll,
                           bool useHeuristic)
      : OpRewritePattern<ModuleOp>(context), lowerAll(lowerAll),
        useHeuristic(useHeuristic) {}

  LogicalResult matchAndRewrite(ModuleOp moduleOp,
                                PatternRewriter &rewriter) const override {
    Operation *parentOp = moduleOp->getParentOp();
    if (!parentOp || !parentOp->getParentOp())
      return failure();

    bool hasClassTag = moduleOp->hasAttr("emitc.class");
    bool meetsHeuristic = false;
    if (useHeuristic && !hasClassTag)
      meetsHeuristic = checkHeuristic(moduleOp);

    if (!lowerAll && !hasClassTag && !meetsHeuristic)
      return failure();

    auto className = moduleOp.getSymName().value_or("class");
    ClassOp classOp = ClassOp::create(rewriter, moduleOp.getLoc(), className);

    Block *classBlock = rewriter.createBlock(&classOp.getBody());

    auto &ops = moduleOp.getBody()->getOperations();
    for (auto &op : llvm::make_early_inc_range(ops)) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      op.moveBefore(classBlock, classBlock->end());
    }

    SmallVector<GlobalOp> globalsToReplace;
    for (auto globalOp : classBlock->getOps<GlobalOp>())
      globalsToReplace.push_back(globalOp);

    for (auto globalOp : globalsToReplace) {
      rewriter.setInsertionPoint(globalOp);
      FieldOp fieldOp = FieldOp::create(rewriter, globalOp.getLoc(),
                                        globalOp.getSymNameAttr(),
                                        globalOp.getTypeAttr(),
                                        globalOp.getInitialValueAttr());

      fieldOp->setDiscardableAttrs(globalOp->getDiscardableAttrDictionary());

      classBlock->walk([&](GetGlobalOp getGlobalOp) {
        if (getGlobalOp.getName() == globalOp.getSymName()) {
          rewriter.setInsertionPoint(getGlobalOp);
          GetFieldOp getFieldOp = GetFieldOp::create(
              rewriter, getGlobalOp.getLoc(), getGlobalOp.getType(),
              getGlobalOp.getNameAttr());
          rewriter.replaceOp(getGlobalOp, getFieldOp);
        }
      });
      rewriter.eraseOp(globalOp);
    }

    rewriter.eraseOp(moduleOp);

    return success();
  }

private:
  bool checkHeuristic(ModuleOp moduleOp) const {
    llvm::DenseSet<StringRef> globalNames;
    moduleOp.walk([&](GlobalOp globalOp) {
      globalNames.insert(globalOp.getSymName());
    });

    if (globalNames.empty())
      return false;

    bool hasMethodUsingGlobal = false;
    moduleOp.walk([&](FuncOp funcOp) {
      funcOp.walk([&](GetGlobalOp getGlobalOp) {
        if (globalNames.contains(getGlobalOp.getName())) {
          hasMethodUsingGlobal = true;
        }
      });
    });

    return hasMethodUsingGlobal;
  }

  bool lowerAll;
  bool useHeuristic;
};

void mlir::emitc::populateLowerNestedModuleToClassPatterns(
    RewritePatternSet &patterns, bool lowerAll, bool useHeuristic) {
  patterns.add<LowerNestedModuleToClass>(patterns.getContext(), lowerAll,
                                         useHeuristic);
}
