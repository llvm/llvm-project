//===- ACCLoopCollapse.cpp - Collapse multi-dimensional parallel loops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass collapses each multi-dimensional scf.parallel loop inside an
// acc.compute_region into a one-dimensional scf.parallel loop. The original
// induction variables are recovered from the collapsed one with remainder and
// division operations. Loops outside an acc.compute_region are left alone.
//
// The collapsed loop keeps every discardable attribute of the original loop
// (for example acc.par_dims, acc.gpu_block_redundant and acc.chunk_size), and
// reductions are carried over by the collapsing utility. The pass also sets
// acc.collapse_count to the number of collapsed dimensions, which remark
// emission reads to report the collapse depth.
//
// Example:
//
//   scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c20) step (%c1, %c1) {
//     ...
//   } {acc.par_dims = #acc<par_dims[block_x]>}
//
// becomes
//
//   scf.parallel (%ij) = (%c0) to (%c200) step (%c1) {
//     %j = arith.remsi %ij, %c20
//     %i = arith.divsi %ij, %c20
//     ...
//   } {acc.collapse_count = 2 : i64, acc.par_dims = #acc<par_dims[block_x]>}
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include <vector>

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCLOOPCOLLAPSE
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-loop-collapse"

using namespace mlir;

namespace {
/// Collapse every dimension of \p parallelOp into a single dimension and carry
/// the attributes of the original loop over to the collapsed loop.
void collapseParallelOp(IRRewriter &rewriter, scf::ParallelOp parallelOp) {
  unsigned collapseDepth = parallelOp.getNumLoops();

  LLVM_DEBUG(llvm::dbgs() << "Collapsing loop with " << collapseDepth
                          << " dimensions\n");

  // A single group containing every dimension of the loop.
  std::vector<unsigned> combinedDims;
  combinedDims.reserve(collapseDepth);
  for (unsigned i = 0; i < collapseDepth; ++i)
    combinedDims.push_back(i);

  // The collapsing utility builds a new loop and drops the attributes of the
  // original one, so save them here.
  SmallVector<NamedAttribute> attrs(parallelOp->getDiscardableAttrs().begin(),
                                    parallelOp->getDiscardableAttrs().end());

  scf::ParallelOp collapsedOp =
      collapseParallelLoops(rewriter, parallelOp, combinedDims);

  for (NamedAttribute attr : attrs)
    collapsedOp->setDiscardableAttr(attr.getName(), attr.getValue());

  // Record the collapse depth so that remarks can report it.
  acc::setCollapseCountAttr(collapsedOp, collapseDepth);
}

class ACCLoopCollapse : public acc::impl::ACCLoopCollapseBase<ACCLoopCollapse> {
public:
  using ACCLoopCollapseBase<ACCLoopCollapse>::ACCLoopCollapseBase;

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Enter ACCLoopCollapse()\n");

    IRRewriter rewriter(&getContext());

    // Only loops inside a compute region are collapsed. The inner walk is
    // post-order, so a nested loop is collapsed before its parent. Replacing
    // the visited loop is safe because the walk has already moved past it.
    getOperation().walk([&](acc::ComputeRegionOp computeRegion) {
      computeRegion.walk([&](scf::ParallelOp op) {
        if (op.getNumLoops() > 1)
          collapseParallelOp(rewriter, op);
      });
    });

    LLVM_DEBUG(llvm::dbgs() << "Exit ACCLoopCollapse()\n");
  }
};
} // namespace
