//===- TileReducerLinalgTiling.cpp - Milestone 10 ---------------*- C++ -*-===//
//
// Tile Linalg reductions via TilingInterface + scf::tileUsingSCF.
// Introduces scf.for. Does not map to GPU threads.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/TileReducerPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::tr {
#define GEN_PASS_DEF_TILETRLINALG
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

static bool isReductionGeneric(linalg::GenericOp op) {
  return llvm::any_of(op.getIteratorTypesArray(), [](utils::IteratorType t) {
    return t == utils::IteratorType::reduction;
  });
}

struct TileTRLinalg : impl::TileTRLinalgBase<TileTRLinalg> {
  using impl::TileTRLinalgBase<TileTRLinalg>::TileTRLinalgBase;

  void runOnOperation() override {
    if (tileSizes.empty())
      return;

    func::FuncOp func = getOperation();
    SmallVector<linalg::GenericOp> targets;
    func.walk([&](linalg::GenericOp op) {
      if (isReductionGeneric(op))
        targets.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp op : targets) {
      auto tilingOp = cast<TilingInterface>(op.getOperation());
      unsigned numLoops = tilingOp.getLoopIteratorTypes().size();

      SmallVector<OpFoldResult> sizes;
      SmallVector<unsigned> reductionDims;
      SmallVector<utils::IteratorType> iters = op.getIteratorTypesArray();
      for (unsigned i = 0; i < numLoops; ++i) {
        int64_t sz = i < tileSizes.size() ? tileSizes[i] : 0;
        sizes.push_back(rewriter.getIndexAttr(sz));
        if (sz != 0 && i < iters.size() &&
            iters[i] == utils::IteratorType::reduction)
          reductionDims.push_back(i);
      }

      scf::SCFTilingOptions options;
      options.setTileSizes(sizes);
      if (!reductionDims.empty())
        options.setReductionDims(reductionDims);

      rewriter.setInsertionPoint(op);
      FailureOr<scf::SCFTilingResult> tiled =
          scf::tileUsingSCF(rewriter, tilingOp, options);
      if (failed(tiled)) {
        op.emitError("failed to tile Linalg reduction");
        return signalPassFailure();
      }
      rewriter.replaceOp(op, tiled->replacements);
    }
  }
};

} // namespace
} // namespace mlir::tr
