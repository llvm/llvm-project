//===- VectorToGPUUtils.cpp - Shared vector-to-MMA helpers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToGPUCommon/VectorToGPUUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include <cstddef>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "vector-to-gpu-common"

using namespace mlir;

bool mlir::constantSupportsMMAMatrixType(arith::ConstantOp constantOp) {
  auto vecType = dyn_cast<VectorType>(constantOp.getType());
  if (!vecType || vecType.getRank() != 2)
    return false;
  return isa<SplatElementsAttr>(constantOp.getValue());
}

bool mlir::broadcastSupportsMMAMatrixType(vector::BroadcastOp broadcastOp) {
  return broadcastOp.getResultVectorType().getRank() == 2;
}

template <typename ExtOpTy>
bool mlir::integerExtendSupportsMMAMatrixType(ExtOpTy extOp) {
  auto transferReadOp =
      extOp.getOperand().template getDefiningOp<vector::TransferReadOp>();
  if (!transferReadOp)
    return false;
  return llvm::all_of(extOp->getUsers(), llvm::IsaPred<vector::ContractionOp>);
}

template bool
    mlir::integerExtendSupportsMMAMatrixType<arith::ExtSIOp>(arith::ExtSIOp);

template bool
    mlir::integerExtendSupportsMMAMatrixType<arith::ExtUIOp>(arith::ExtUIOp);

std::optional<gpu::MMAElementwiseOp>
mlir::convertElementwiseOpToMMA(Operation *op) {
  using MMAEwO = gpu::MMAElementwiseOp;
  return TypeSwitch<Operation *, std::optional<MMAEwO>>(op)
      .Case([](arith::AddFOp) { return MMAEwO::ADDF; })
      .Case([](arith::AddIOp) { return MMAEwO::ADDI; })
      .Case([](arith::DivFOp) { return MMAEwO::DIVF; })
      .Case([](arith::DivSIOp) { return MMAEwO::DIVS; })
      .Case([](arith::DivUIOp) { return MMAEwO::DIVU; })
      .Case([](arith::ExtFOp) { return MMAEwO::EXTF; })
      .Case([](arith::MaximumFOp) { return MMAEwO::MAXF; })
      .Case([](arith::MinimumFOp) { return MMAEwO::MINF; })
      .Case([](arith::MulFOp) { return MMAEwO::MULF; })
      .Case([](arith::MulIOp) { return MMAEwO::MULI; })
      .Case([](arith::NegFOp) { return MMAEwO::NEGATEF; })
      .Case([](arith::SubFOp) { return MMAEwO::SUBF; })
      .Case([](arith::SubIOp) { return MMAEwO::SUBI; })
      .Case([](arith::TruncFOp) { return MMAEwO::TRUNCF; })
      .Default(std::nullopt);
}

bool mlir::elementwiseSupportsMMAMatrixType(Operation *op) {
  return convertElementwiseOpToMMA(op).has_value();
}

// Replace ForOp with a new ForOp with extra operands. The YieldOp is not
// updated and needs to be updated separately for the loop to be correct.
static scf::ForOp replaceForOpWithNewSignature(RewriterBase &rewriter,
                                               scf::ForOp loop,
                                               ValueRange newInitArgs) {
  OpBuilder::InsertionGuard g(rewriter);

  // Create a new loop before the existing one, with the extra operands.
  rewriter.setInsertionPoint(loop);
  auto operands = llvm::to_vector<4>(loop.getInitArgs());
  llvm::append_range(operands, newInitArgs);
  scf::ForOp newLoop =
      scf::ForOp::create(rewriter, loop.getLoc(), loop.getLowerBound(),
                         loop.getUpperBound(), loop.getStep(), operands);
  rewriter.eraseBlock(newLoop.getBody());

  newLoop.getRegion().getBlocks().splice(
      newLoop.getRegion().getBlocks().begin(), loop.getRegion().getBlocks());
  for (Value operand : newInitArgs)
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());

  for (auto it : llvm::zip(loop.getResults(), newLoop.getResults().take_front(
                                                  loop.getNumResults())))
    rewriter.replaceAllUsesWith(std::get<0>(it), std::get<1>(it));

  LDBG() << "newLoop now: " << newLoop;
  LDBG() << "stripped scf.for: " << loop;
  LDBG() << "erase: " << loop;

  rewriter.eraseOp(loop);
  return newLoop;
}

LogicalResult
mlir::convertForOpForMMA(RewriterBase &rewriter, scf::ForOp op,
                         llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  SmallVector<Value> newOperands;
  SmallVector<std::pair<size_t, size_t>> argMapping;
  for (const auto &operand : llvm::enumerate(op.getInitArgs())) {
    auto it = valueMapping.find(operand.value());
    if (it == valueMapping.end()) {
      LDBG() << "no value mapping for: " << operand.value();
      continue;
    }
    argMapping.push_back(std::make_pair(
        operand.index(), op.getInitArgs().size() + newOperands.size()));
    newOperands.push_back(it->second);
  }

  scf::ForOp newForOp = replaceForOpWithNewSignature(rewriter, op, newOperands);
  Block &loopBody = *newForOp.getBody();
  for (auto mapping : argMapping) {
    valueMapping[newForOp.getResult(mapping.first)] =
        newForOp.getResult(mapping.second);
    valueMapping[loopBody.getArgument(mapping.first +
                                      newForOp.getNumInductionVars())] =
        loopBody.getArgument(mapping.second + newForOp.getNumInductionVars());
  }

  LDBG() << "scf.for to: " << newForOp;
  return success();
}

LogicalResult
mlir::convertYieldOpForMMA(RewriterBase &rewriter, scf::YieldOp op,
                           llvm::DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  auto loop = cast<scf::ForOp>(op->getParentOp());
  auto yieldOperands = llvm::to_vector<4>(op.getOperands());
  for (const auto &operand : llvm::enumerate(op.getOperands())) {
    auto it = valueMapping.find(operand.value());
    if (it == valueMapping.end())
      continue;
    // Replace the yield of old value with the for op argument to make it easier
    // to remove the dead code.
    yieldOperands[operand.index()] = loop.getInitArgs()[operand.index()];
    yieldOperands.push_back(it->second);
  }
  scf::YieldOp::create(rewriter, op.getLoc(), yieldOperands);

  LDBG() << "erase: " << op;
  rewriter.eraseOp(op);
  return success();
}

namespace {
// Fold vector.transpose into vector.transfer_read, including through
// arithmetic extension operations.
struct CombineTransferReadOpTranspose final
    : public OpRewritePattern<vector::TransposeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // Look through integer extend ops.
    Value source = op.getVector();
    Type resultType = op.getType();
    Operation *extOp;
    if ((extOp = source.getDefiningOp<arith::ExtSIOp>()) ||
        (extOp = source.getDefiningOp<arith::ExtUIOp>()) ||
        (extOp = source.getDefiningOp<arith::ExtFOp>())) {
      source = extOp->getOperand(0);
      resultType =
          VectorType::get(cast<VectorType>(resultType).getShape(),
                          cast<VectorType>(source.getType()).getElementType());
    }

    auto transferReadOp = source.getDefiningOp<vector::TransferReadOp>();
    if (!transferReadOp)
      return rewriter.notifyMatchFailure(op, "no transfer read");

    // TODO: support 0-d corner case.
    if (transferReadOp.getTransferRank() == 0)
      return rewriter.notifyMatchFailure(op, "0-D transfer read");

    if (transferReadOp.getMask() || transferReadOp.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(op, "not inbounds transfer read");

    AffineMap permutationMap =
        AffineMap::getPermutationMap(op.getPermutation(), op.getContext());
    AffineMap newMap =
        permutationMap.compose(transferReadOp.getPermutationMap());

    auto loc = op.getLoc();
    Value result = vector::TransferReadOp::create(
                       rewriter, loc, resultType, transferReadOp.getBase(),
                       transferReadOp.getIndices(), AffineMapAttr::get(newMap),
                       transferReadOp.getPadding(), transferReadOp.getMask(),
                       transferReadOp.getInBoundsAttr())
                       .getResult();

    // Fuse through the integer extend op.
    if (extOp) {
      if (isa<arith::ExtSIOp>(extOp))
        result = arith::ExtSIOp::create(rewriter, loc, op.getType(), result)
                     .getResult();
      else if (isa<arith::ExtUIOp>(extOp))
        result = arith::ExtUIOp::create(rewriter, loc, op.getType(), result)
                     .getResult();
      else
        result = arith::ExtFOp::create(rewriter, loc, TypeRange{op.getType()},
                                       ValueRange{result},
                                       arith::ExtFOp::Properties{})
                     .getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::populateVectorTransferReadTransposePatterns(
    RewritePatternSet &patterns) {
  patterns.add<CombineTransferReadOpTranspose>(patterns.getContext());
}
