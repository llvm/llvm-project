//===- RewriteInsertsPass.cpp - MLIR conversion pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to rewrite sequential chains of
// `spirv::CompositeInsert` operations into `spirv::CompositeConstruct`
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVREWRITEINSERTSPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;

namespace {

/// Replaces sequential chains of `spirv::CompositeInsertOp` operation into
/// `spirv::CompositeConstructOp` operation if possible.
class RewriteInsertsPass
    : public spirv::impl::SPIRVRewriteInsertsPassBase<RewriteInsertsPass> {
public:
  void runOnOperation() override;

private:
  /// Collects a sequential insertion chain by the given
  /// `spirv::CompositeInsertOp` operation, if the given operation is the last
  /// in the chain.
  LogicalResult
  collectInsertionChain(spirv::CompositeInsertOp op,
                        SmallVectorImpl<spirv::CompositeInsertOp> &insertions);
};

} // namespace

void RewriteInsertsPass::runOnOperation() {
  SmallVector<SmallVector<spirv::CompositeInsertOp, 4>, 4> workList;
  getOperation().walk([this, &workList](spirv::CompositeInsertOp op) {
    SmallVector<spirv::CompositeInsertOp, 4> insertions;
    if (succeeded(collectInsertionChain(op, insertions)))
      workList.push_back(insertions);
  });

  for (const auto &insertions : workList) {
    auto lastCompositeInsertOp = insertions.back();
    auto compositeType = lastCompositeInsertOp.getType();
    auto location = lastCompositeInsertOp.getLoc();

    SmallVector<Value, 4> operands;
    // Collect inserted objects.
    for (auto insertionOp : insertions)
      operands.push_back(insertionOp.getObject());

    OpBuilder builder(lastCompositeInsertOp);
    auto compositeConstructOp = builder.create<spirv::CompositeConstructOp>(
        location, compositeType, operands);

    lastCompositeInsertOp.replaceAllUsesWith(
        compositeConstructOp->getResult(0));

    // Erase ops.
    for (auto insertOp : llvm::reverse(insertions)) {
      auto *op = insertOp.getOperation();
      if (op->use_empty())
        insertOp.erase();
    }
  }
}

LogicalResult RewriteInsertsPass::collectInsertionChain(
    spirv::CompositeInsertOp op,
    SmallVectorImpl<spirv::CompositeInsertOp> &insertions) {
  if (isa<spirv::CooperativeMatrixType>(op.getComposite().getType()))
    return failure();

  auto indicesArrayAttr = cast<ArrayAttr>(op.getIndices());
  // TODO: handle nested composite object.
  if (indicesArrayAttr.size() == 1) {
    auto numElements = cast<spirv::CompositeType>(op.getComposite().getType())
                           .getNumElements();

    auto index = cast<IntegerAttr>(indicesArrayAttr[0]).getInt();
    // Need a last index to collect a sequential chain.
    if (index + 1 != numElements)
      return failure();

    insertions.resize(numElements);
    while (true) {
      insertions[index] = op;

      if (index == 0)
        return success();

      op = op.getComposite().getDefiningOp<spirv::CompositeInsertOp>();
      if (!op)
        return failure();

      --index;
      indicesArrayAttr = cast<ArrayAttr>(op.getIndices());
      if ((indicesArrayAttr.size() != 1) ||
          (cast<IntegerAttr>(indicesArrayAttr[0]).getInt() != index))
        return failure();
    }
  }
  return failure();
}
