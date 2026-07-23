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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_ELEVATEALLOCSTOGLOBALSPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

/// Returns true if `op` is contained inside any branching, region, or looping
/// structure (such as scf.for, scf.if, or repetitive regions)
static bool isInsideControlFlow(Operation *op) {
  return getEnclosingRepetitiveRegion(op) ||
         op->getParentOfType<LoopLikeOpInterface>() ||
         op->getParentOfType<RegionBranchOpInterface>();
}

/// Elevates a static `memref.alloc` operation to a top-level `memref.global` op
/// if the allocation is not enclosed within any control flow constructs.
///
/// Converts:
/// ```mlir
/// %0 = memref.alloc() : memref<4x4xf32>
/// memref.dealloc %0 : memref<4x4xf32>
/// ```
/// to:
/// ```mlir
/// memref.global "private" @global_alloc : memref<4x4xf32>
/// ...
/// %0 = memref.get_global @global_alloc : memref<4x4xf32>
/// ```
struct ElevateAllocsToGlobals : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const final {
    auto memrefType = allocOp.getType();
    // `memref.global` requires statically shaped memrefs with no dynamic sizes.
    if (!memrefType.hasStaticShape() || !allocOp.getDynamicSizes().empty())
      return failure();

    // Avoid elevating allocations inside control flow (loops or conditionals),
    // as converting them to a single static global would make multiple
    // executions share the same buffer, changing semantics or causing race
    // conditions.
    if (isInsideControlFlow(allocOp))
      return failure();

    // Create the global variable at the nearest enclosing symbol table defining
    // op if it's a ModuleOp.
    auto moduleOp = llvm::dyn_cast_or_null<ModuleOp>(
        SymbolTable::getNearestSymbolTable(allocOp));
    if (!moduleOp)
      return failure();

    OpBuilder detachedBuilder(rewriter.getContext());
    StringAttr globalName = rewriter.getStringAttr("global_alloc");
    memref::GlobalOp globalOp = memref::GlobalOp::create(
        detachedBuilder, allocOp.getLoc(), globalName,
        rewriter.getStringAttr("private"), memrefType, rewriter.getUnitAttr(),
        false, allocOp.getAlignmentAttr());

    SymbolTable(moduleOp).insert(globalOp);

    // Remove any `memref.dealloc` operations using this allocation
    SmallVector<Operation *> deallocsToDelete;
    for (OpOperand &use : allocOp.getResult().getUses()) {
      Operation *user = use.getOwner();
      if (isa<memref::DeallocOp>(user))
        deallocsToDelete.push_back(user);
    }
    for (Operation *dealloc : deallocsToDelete)
      rewriter.eraseOp(dealloc);

    // Replace the original `memref.alloc` with `memref.get_global`.
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
