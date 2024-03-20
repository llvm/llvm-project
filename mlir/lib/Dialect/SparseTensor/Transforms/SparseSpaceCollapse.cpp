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

namespace sparse_tensor {

bool isCollapsableIterations(LoopLikeOpInterface parent,
                             LoopLikeOpInterface node) {
  auto pIterArgs = parent.getRegionIterArgs();
  auto nInitArgs = node.getInits();
  if (pIterArgs.size() != nInitArgs.size())
    return false;

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

bool legalToCollapse(ExtractIterSpaceOp parent, ExtractIterSpaceOp node) {
  auto pItOp = llvm::dyn_cast<IterateOp>(parent->getParentOp());
  auto nItOp = llvm::dyn_cast<IterateOp>(node->getParentOp());

  // Can only collapse spaces extracted from the same tensor.
  if (parent.getTensor() != node.getTensor() || !parent->hasOneUse())
    return false;

  // Can only collapse consecutive simple iteration on one tensor (i.e., no
  // coiteration).
  if (!nItOp || nItOp.getIterSpace() != parent.getResult() ||
      nItOp->getBlock() != parent->getBlock())
    return false;

  if (pItOp && !isCollapsableIterations(pItOp, nItOp))
    return false;

  // TODO: Make sure all other operations in the same basic block as `node` can
  // be collapsed and sink into the collapsed iteration (through Interfaces
  // defined in TD files).
  return true;
}

void collapseSparseSpace(ArrayRef<ExtractIterSpaceOp> toCollapse) {
  if (toCollapse.size() < 2)
    return;

  ExtractIterSpaceOp root = toCollapse.front();
  ExtractIterSpaceOp leaf = toCollapse.back();
  Location loc = root.getLoc();

  if (!leaf->hasOneUse())
    return;
  assert(root->hasOneUse());

  // Insert collapsed operation at the same scope as root operation.
  OpBuilder builder(toCollapse.front());

  // Construct the collapsed iteration space.
  auto collapsedSpace = builder.create<ExtractIterSpaceOp>(
      loc, root.getTensor(), root.getParentIter(), root.getLoLvl(),
      leaf.getHiLvl());

  auto rItOp = llvm::cast<IterateOp>(*root->getUsers().begin());
  auto pItOp = llvm::cast<IterateOp>(leaf->getParentOp());

  // This could either be IterateOp or (TODO: in the future) CoIterateOp.
  auto loop = llvm::dyn_cast<IterateOp>(*leaf->getUsers().begin());
  if (!loop || !isCollapsableIterations(pItOp, loop))
    return;

  IRMapping mapper;
  mapper.map(leaf, collapsedSpace.getResultSpace());
  for (auto z : llvm::zip_equal(loop.getInitArgs(), rItOp.getInitArgs()))
    mapper.map(std::get<0>(z), std::get<1>(z));

  auto cloned = llvm::cast<IterateOp>(builder.clone(*loop, mapper));
  cloned.getIterator().setType(collapsedSpace.getType().getIteratorType());

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
    SmallVector<ExtractIterSpaceOp> toCollapse;
    func->walk([&](ExtractIterSpaceOp op) {
      if (toCollapse.empty()) {
        // Root space to collapse.
        toCollapse.push_back(op);
      } else {
        if (legalToCollapse(toCollapse.back(), op)) {
          toCollapse.push_back(op);
        } else {
          collapseSparseSpace(toCollapse);
          toCollapse.clear();
        }
      }
    });

    collapseSparseSpace(toCollapse);
  }
};

} // namespace sparse_tensor

std::unique_ptr<Pass> createSparseSpaceCollapsePass() {
  return std::make_unique<sparse_tensor::SparseSpaceCollapsePass>();
}

} // namespace mlir
