//===--------- SparseSpaceCollapse.cpp - Collapse Sparse Space Pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_SPARSESPACECOLLAPSE
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "sparse-space-collapse"

using namespace mlir;
using namespace sparse_tensor;

namespace {

struct CollapseSpaceInfo {
  ExtractIterSpaceOp space;
  IterateOp loop;
};

bool isCollapsableLoops(LoopLikeOpInterface parent, LoopLikeOpInterface node) {
  auto pIterArgs = parent.getRegionIterArgs();
  auto nInitArgs = node.getInits();
  if (pIterArgs.size() != nInitArgs.size())
    return false;

  // Two loops are collapsable if they are perfectly nested.
  auto pYields = parent.getYieldedValues();
  auto nResult = node.getLoopResults().value();

  bool yieldEq =
      llvm::all_of(llvm::zip_equal(pYields, nResult), [](auto zipped) {
        return std::get<0>(zipped) == std::get<1>(zipped);
      });

  // Parent iter_args should be passed directly to the node's init_args.
  bool iterArgEq =
      llvm::all_of(llvm::zip_equal(pIterArgs, nInitArgs), [](auto zipped) {
        return std::get<0>(zipped) == std::get<1>(zipped);
      });

  return yieldEq && iterArgEq;
}

bool legalToCollapse(SmallVectorImpl<CollapseSpaceInfo> &toCollapse,
                     ExtractIterSpaceOp curSpace) {

  auto getIterateOpOverSpace = [](ExtractIterSpaceOp space) -> IterateOp {
    Value spaceVal = space.getExtractedSpace();
    if (spaceVal.hasOneUse())
      return llvm::dyn_cast<IterateOp>(*spaceVal.getUsers().begin());
    return nullptr;
  };

  if (toCollapse.empty()) {
    // Collapse root.
    if (auto itOp = getIterateOpOverSpace(curSpace)) {
      CollapseSpaceInfo &info = toCollapse.emplace_back();
      info.space = curSpace;
      info.loop = itOp;
      return true;
    }
    return false;
  }

  auto parent = toCollapse.back().space;
  auto pItOp = toCollapse.back().loop;
  auto nItOp = getIterateOpOverSpace(curSpace);

  // Can only collapse spaces extracted from the same tensor.
  if (parent.getTensor() != curSpace.getTensor()) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "failed to collpase spaces extracted from different tensors.";
    });
    return false;
  }

  // Can only collapse consecutive simple iteration on one tensor (i.e., no
  // coiteration).
  if (!nItOp || nItOp->getBlock() != curSpace->getBlock() ||
      pItOp.getIterator() != curSpace.getParentIter() ||
      curSpace->getParentOp() != pItOp.getOperation()) {
    LLVM_DEBUG(
        { llvm::dbgs() << "failed to collapse non-consecutive IterateOps."; });
    return false;
  }

  if (pItOp && !isCollapsableLoops(pItOp, nItOp)) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "failed to collapse IterateOps that are not perfectly nested.";
    });
    return false;
  }

  CollapseSpaceInfo &info = toCollapse.emplace_back();
  info.space = curSpace;
  info.loop = nItOp;
  return true;
}

void collapseSparseSpace(MutableArrayRef<CollapseSpaceInfo> toCollapse) {
  if (toCollapse.size() < 2)
    return;

  ExtractIterSpaceOp root = toCollapse.front().space;
  ExtractIterSpaceOp leaf = toCollapse.back().space;
  Location loc = root.getLoc();

  assert(root->hasOneUse() && leaf->hasOneUse());

  // Insert collapsed operation at the same scope as root operation.
  OpBuilder builder(root);

  // Construct the collapsed iteration space.
  auto collapsedSpace = builder.create<ExtractIterSpaceOp>(
      loc, root.getTensor(), root.getParentIter(), root.getLoLvl(),
      leaf.getHiLvl());

  auto rItOp = llvm::cast<IterateOp>(*root->getUsers().begin());
  auto innermost = toCollapse.back().loop;

  IRMapping mapper;
  mapper.map(leaf, collapsedSpace.getExtractedSpace());
  for (auto z : llvm::zip_equal(innermost.getInitArgs(), rItOp.getInitArgs()))
    mapper.map(std::get<0>(z), std::get<1>(z));

  auto cloned = llvm::cast<IterateOp>(builder.clone(*innermost, mapper));
  builder.setInsertionPointToStart(cloned.getBody());

  LevelSet crdUsedLvls;
  unsigned shift = 0, argIdx = 1;
  for (auto info : toCollapse.drop_back()) {
    LevelSet set = info.loop.getCrdUsedLvls();
    crdUsedLvls |= set.lshift(shift);
    shift += info.loop.getSpaceDim();
    for (BlockArgument crd : info.loop.getCrds()) {
      BlockArgument collapsedCrd = cloned.getBody()->insertArgument(
          argIdx++, builder.getIndexType(), crd.getLoc());
      crd.replaceAllUsesWith(collapsedCrd);
    }
  }
  crdUsedLvls |= innermost.getCrdUsedLvls().lshift(shift);
  cloned.getIterator().setType(collapsedSpace.getType().getIteratorType());
  cloned.setCrdUsedLvls(crdUsedLvls);

  rItOp.replaceAllUsesWith(cloned.getResults());
  // Erase collapsed loops.
  rItOp.erase();
  root.erase();
}

struct SparseSpaceCollapsePass
    : public impl::SparseSpaceCollapseBase<SparseSpaceCollapsePass> {
  SparseSpaceCollapsePass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // A naive (experimental) implementation to collapse consecutive sparse
    // spaces. It does NOT handle complex cases where multiple spaces are
    // extracted in the same basic block. E.g.,
    //
    // %space1 = extract_space %t1 ...
    // %space2 = extract_space %t2 ...
    // sparse_tensor.iterate(%sp1) ...
    //
    SmallVector<CollapseSpaceInfo> toCollapse;
    func->walk([&](ExtractIterSpaceOp op) {
      if (!legalToCollapse(toCollapse, op)) {
        // if not legal to collapse one more space, collapse the existing ones
        // and clear.
        collapseSparseSpace(toCollapse);
        toCollapse.clear();
      }
    });

    collapseSparseSpace(toCollapse);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createSparseSpaceCollapsePass() {
  return std::make_unique<SparseSpaceCollapsePass>();
}
