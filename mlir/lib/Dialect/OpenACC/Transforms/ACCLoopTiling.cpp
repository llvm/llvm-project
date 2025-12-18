//===- ACCLoopTiling.cpp - Tile ACC Loops ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements the OpenACC loop tiling transformation for acc.loop
// operations that have the tile clause (OpenACC 3.4 spec, section 2.9.8).
//
// Overview:
// ---------
// The tile clause specifies that the iterations of the associated loops should
// be divided into tiles (rectangular blocks). This pass transforms a single
// or nested acc.loop with tile clauses into a structure of "tile loops"
// (iterating over tiles) containing "element loops" (iterating within tiles).
//
// For example, tiling a 2-level nested loop with tile(T1, T2) produces:
//
//   // Before tiling:
//   acc.loop tile(T1, T2) control(%i, %j) = (lb1, lb2) to (ub1, ub2) step (s1,
//   s2)
//
//   // After tiling:
//   acc.loop control(%i) = (lb1) to (ub1) step (s1*T1) {      // tile loop 1
//     acc.loop control(%j) = (lb2) to (ub2) step (s2*T2) {    // tile loop 2
//       acc.loop control(%ii) = (%i) to (min(ub1, %i+s1*T1)) step (s1) { //
//       element 1
//         acc.loop control(%jj) = (%j) to (min(ub2, %j+s2*T2)) step (s2) { //
//         element 2
//           // loop body using %ii, %jj
//         }
//       }
//     }
//   }
//
// Gang/worker/vector attributes are distributed as follows:
// - gang: applied to tile loops
// - vector: applied to element loops
// - worker: removed from inner loops
//
// Unknown Tile Sizes:
// -------------------
// The OpenACC tile(*) syntax indicates an implementation-defined tile size.
// In the IR, this is represented as -1. The pass resolves these to the
// default tile size (configurable via pass option).
//
// Requirements:
// -------------
// 1. The pass uses the OpenACCSupport analysis for remark and NYI (not yet
//    implemented) emission. Custom implementations can be registered via
//    setImplementation() to provide pipeline-specific handling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsTiling.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCLOOPTILING
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-loop-tile"

namespace {
using namespace mlir;

struct ACCLoopTilingImpl : public OpRewritePattern<acc::LoopOp> {
  ACCLoopTilingImpl(MLIRContext *context, int32_t defaultTileSize,
                    acc::OpenACCSupport &accSupport)
      : OpRewritePattern<acc::LoopOp>(context),
        defaultTileSize(defaultTileSize), accSupport(accSupport) {}

  // Check that tile size types are not narrower than IV types.
  // We only check when both types are IntegerType. For IndexType, the width
  // is target-dependent and the casting utility will handle it correctly.
  LogicalResult checkTileSizeTypes(acc::LoopOp loop,
                                   ArrayRef<Value> tileSizes) const {
    auto ivTypes = loop.getBody().getArgumentTypes();
    for (size_t i = 0; i < tileSizes.size() && i < ivTypes.size(); ++i) {
      Type tileType = tileSizes[i].getType();
      Type ivType = ivTypes[i];

      // Skip unknown tile sizes (will be created with correct type)
      auto constVal = getConstantIntValue(tileSizes[i]);
      if (constVal && *constVal < 0)
        continue;

      // Only compare when both are integer types (not index)
      auto tileIntType = dyn_cast<IntegerType>(tileType);
      auto ivIntType = dyn_cast<IntegerType>(ivType);
      if (tileIntType && ivIntType) {
        if (tileIntType.getWidth() > ivIntType.getWidth()) {
          accSupport.emitNYI(loop.getLoc(),
                             "tile size type (i" +
                                 std::to_string(tileIntType.getWidth()) +
                                 ") is wider than loop IV type (i" +
                                 std::to_string(ivIntType.getWidth()) + ")");
          return failure();
        }
      }
    }
    return success();
  }

  void emitTilingRemarks(acc::LoopOp loop, ArrayRef<Value> tileSizes) const {
    // Emit remarks for loop tiling
    size_t tileLevel = tileSizes.size();
    std::string msg =
        "Tiling " + std::to_string(tileLevel) + "-level loop nest with tile(";
    for (size_t i = 0; i < tileSizes.size(); ++i) {
      std::optional<int64_t> val = getConstantIntValue(tileSizes[i]);
      if (*val == -1)
        msg += "*";
      else
        msg += std::to_string(*val);
      if (i < tileSizes.size() - 1)
        msg += ",";
    }
    msg += ")";
    accSupport.emitRemark(loop, llvm::Twine(msg), DEBUG_TYPE);

    // Emit remarks for unknown tile sizes that will be resolved to default
    // TODO: Need to base the default tile size on some heuristics.
    for (Value tileSize : tileSizes) {
      std::optional<int64_t> val = getConstantIntValue(tileSize);
      if (val && *val < 0) {
        std::string unknownMsg = "Picking default tile size " +
                                 std::to_string(defaultTileSize) +
                                 " for unknown tile size '*'";
        accSupport.emitRemark(loop, llvm::Twine(unknownMsg), DEBUG_TYPE);
      }
    }
  }

  LogicalResult matchAndRewrite(acc::LoopOp origLoop,
                                PatternRewriter &rewriter) const override {

    if (origLoop.getTileValues().empty())
      return success();

    SmallVector<Value> tileSizes(origLoop.getTileValues().begin(),
                                 origLoop.getTileValues().end());
    unsigned tileCount = tileSizes.size();
    unsigned collapseCount = origLoop.getCollapseValue().value_or(1);

    // Sanity check tile size types
    if (failed(checkTileSizeTypes(origLoop, tileSizes)))
      return failure();

    // Emit remarks for loop tiling. This is emitted before the original loop
    // is modified. However, it assumes that tiling will not fail.
    emitTilingRemarks(origLoop, tileSizes);

    LLVM_DEBUG(llvm::dbgs() << "\nBefore tiling:\n" << *origLoop << "\n");

    // Clear tile operands from origLoop
    rewriter.startOpModification(origLoop);
    origLoop.getTileOperandsMutable().clear();
    origLoop.removeTileOperandsSegmentsAttr();
    origLoop.removeTileOperandsDeviceTypeAttr();
    rewriter.finalizeOpModification(origLoop);

    SmallVector<acc::LoopOp> loopsToTile;
    if (collapseCount < tileCount) {
      // Uncollapse tile loops before tiling if necessary
      loopsToTile =
          acc::uncollapseLoops(origLoop, tileCount, collapseCount, rewriter);
      rewriter.replaceOp(origLoop, loopsToTile[0]);
      LLVM_DEBUG(llvm::dbgs() << "\nAfter uncollapsing:\n"
                              << *loopsToTile[0] << "\n");
    } else {
      loopsToTile.push_back(origLoop);
    }

    // loopsToTile is a vector of perfectly nested loops. The outermost loop
    // may have multiple IVs but inner loops can only have one IV.
    // The utility handles unknown tile sizes (*) by using `defaultTileSize`.
    acc::tileACCLoops(loopsToTile, tileSizes, defaultTileSize, rewriter);

    LLVM_DEBUG(llvm::dbgs() << "\nAfter tiling:\n " << *loopsToTile[0] << "\n");
    return success();
  }

private:
  int32_t defaultTileSize;
  acc::OpenACCSupport &accSupport;
};

class ACCLoopTiling : public acc::impl::ACCLoopTilingBase<ACCLoopTiling> {
public:
  using ACCLoopTilingBase<ACCLoopTiling>::ACCLoopTilingBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();

    RewritePatternSet patterns(context);
    patterns.insert<ACCLoopTilingImpl>(context, defaultTileSize, accSupport);
    GreedyRewriteConfig grc;
    grc.setUseTopDownTraversal(true);
    grc.setMaxIterations(1);
    (void)applyPatternsGreedily(funcOp, std::move(patterns), grc);
  }
};

} // namespace
