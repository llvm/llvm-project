//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_ELEVATEALLOCSTOGLOBALSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

// Checks if 'op' is contained inside any branching or looping structure
static bool isInsideControlFlow(mlir::Operation *op) {
  if (mlir::getEnclosingRepetitiveRegion(op) != nullptr)
    return true;

  if (op->getParentOfType<mlir::LoopLikeOpInterface>())
    return true;

  if (auto regionParent = op->getParentOfType<mlir::RegionBranchOpInterface>())
    return true;

  return false;
}

struct ElevateAllocsToGlobals : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const final {

    auto memrefType = allocOp.getType();
    // memref.global requires statically shaped memrefs
    if (!memrefType.hasStaticShape() || !allocOp.getDynamicSizes().empty())
      return failure();

    auto loopParent = allocOp->getParentOfType<mlir::LoopLikeOpInterface>();
    if (loopParent != nullptr || isInsideControlFlow(allocOp))
      return failure();

    memref::GlobalOp globalOp;
    {
      Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(allocOp);

      SymbolTable symbolTable(symbolTableOp);

      OpBuilder builder(rewriter.getContext());
      StringAttr globalName = rewriter.getStringAttr("global_alloc");
      globalOp = memref::GlobalOp::create(builder, allocOp.getLoc(), globalName,
                                          rewriter.getStringAttr("private"),
                                          memrefType, rewriter.getUnitAttr(),
                                          false, allocOp.getAlignmentAttr());

      symbolTable.insert(globalOp);
    }

    SmallVector<Operation *> deallocsToDelete;
    for (OpOperand &use : allocOp.getResult().getUses()) {
      Operation *user = use.getOwner();
      if (isa<memref::DeallocOp>(user))
        deallocsToDelete.push_back(user);
    }
    for (Operation *dealloc : deallocsToDelete)
      rewriter.eraseOp(dealloc);

    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(allocOp, memrefType,
                                                     globalOp.getName());

    return success();
  }
};

struct ElevateAllocsToGlobalsPass
    : public mlir::memref::impl::ElevateAllocsToGlobalsPassBase<
          ElevateAllocsToGlobalsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    memref::populateElevateAllocsToGlobalsPatterns(patterns);

    (void)applyPatternsGreedily(moduleOp, std::move(patterns));
  }
};
} // namespace

void mlir::memref::populateElevateAllocsToGlobalsPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<ElevateAllocsToGlobals>(patterns.getContext());
}
