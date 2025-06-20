//===- HoistVectorTransfers.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

// Function to retrives vector transfer read operations (Acc, Lhs, and Rhs) from contraction operation. 
static FailureOr<SmallVector<vector::TransferReadOp>>
getContractOperands(vector::ContractionOp contractOp) {
  SmallVector<vector::TransferReadOp> list;
  for (int i = 0; i < 3; i++) {
    auto vectorReadOp =
        contractOp.getOperand(i).getDefiningOp<vector::TransferReadOp>();
    if (!vectorReadOp)
      return failure();
    list.push_back(vectorReadOp);
  }
  return list;
}

// Function to retrive subview from vector transfer read operation.
static FailureOr<SmallVector<memref::SubViewOp>>
getReadOperands(SmallVector<vector::TransferReadOp> readOps) {
  SmallVector<memref::SubViewOp> list;
  for (vector::TransferReadOp readOp : readOps) {
    auto subViewOp = readOp.getOperand(0).getDefiningOp<memref::SubViewOp>();
    if (!subViewOp)
      return failure();
    list.push_back(subViewOp);
  }
  return list;
}

// Function to retrive the tiled nested loop structure (m->n->reduction->k) for the contract operation
static FailureOr<SmallVector<scf::ForOp>>
getNestedLoop(vector::ContractionOp contractOp) {
  SmallVector<scf::ForOp> list;
  Operation *current = contractOp;
  for (int i = 0; i < 4; i++) {
    Operation *parent = current->getParentOfType<scf::ForOp>();
    if (!parent)
      return failure();
    list.push_back(dyn_cast<scf::ForOp>(parent));
    current = parent;
  }
  return list;
}

// Function to check iv of nested loops matches with the subview
static LogicalResult checkNestedLoop(SmallVector<scf::ForOp> loops,
                                     SmallVector<memref::SubViewOp> subviews) {
  auto subviewOpLhsOffsets = subviews[0].getOffsets();
  auto subviewOpRhsOffsets = subviews[1].getOffsets();
  auto subviewOpAccOffsets = subviews[2].getOffsets();

  Value ivK = loops[0].getInductionVar();
  if (ivK != subviewOpLhsOffsets[2] || ivK != subviewOpRhsOffsets[1])
    return failure();

  Value ivReduction = loops[1].getInductionVar();
  if (ivReduction != subviewOpLhsOffsets[0] ||
      ivReduction != subviewOpRhsOffsets[0])
    return failure();

  Value ivN = loops[2].getInductionVar();
  if (ivN != subviewOpAccOffsets[1] || ivN != subviewOpRhsOffsets[2])
    return failure();

  Value ivM = loops[3].getInductionVar();
  if (ivM != subviewOpLhsOffsets[1] || ivM != subviewOpAccOffsets[0])
    return failure();

  return success();
}

/// Hoist vector transfer read and write operations for the tiled batch reduce matmul operation
/// outside the reduction and k-loop.
///
/// As an example, the following pseudo-code will be rewritten
/// scf.for %arg3 = %c0 to %c32 step %c4  // m-loop
///  scf.for %arg4 = %c0 to %c64 step %c64   // n-loop
///   %subview_2 = memref.subview %subview[%arg3, %arg4] [4, 64] [1, 1] 
///    scf.for %arg5 = %c0 to %c24 step %c1   // reduction-loop
///     scf.for %arg6 = %c0 to %c64 step %c1   // k-loop
///      %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg6] [1, 4, 1] [1, 1, 1] 
///      %subview_4 = memref.subview %0[%arg5, %arg6, %arg4] [1, 1, 64] [1, 1, 1] 
///      %1 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} 
///      %2 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} 
///      %3 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} 
///      %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 
///      vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]}
/// to:
/// scf.for %arg3 = %c0 to %c32 step %c4 
///  scf.for %arg4 = %c0 to %c64 step %c64 
///   %subview_2 = memref.subview %subview[%arg3, %arg4] [4, 64] [1, 1] 
///   %1 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} 
///   %2 = scf.for %arg5 = %c0 to %c24 step %c1 iter_args(%arg6 = %1) -> (!type) {
///    %3 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %arg6) -> (!type) {
///     %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg7] [1, 4, 1] [1, 1, 1] 
///     %subview_4 = memref.subview %0[%arg5, %arg7, %arg4] [1, 1, 64] [1, 1, 1] 
///     %4 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} 
///     %5 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} 
///     %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 
///     scf.yield %6 : !type
///    }
///    scf.yield %3 : !type
///   }
///   vector.transfer_write %2, %subview_2[%c0, %c0] {in_bounds = [true, true]} 
///
struct HoistVectorTransferOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    // Check the vector contract operation satisfies the required pattern.
    // Check the Acc, Lhs, and Rhs of contract operation
    auto operands = getContractOperands(contractOp);
    if (failed(operands))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Invalid operands for contract op");

    auto readOps = *operands;
    auto vectorReadOpAcc = readOps[2];
    auto vectorReadOpLhs = readOps[0];
    auto vectorReadOpRhs = readOps[1];

    // Check whether the operand of vector transfer read is a subview
    auto subviews = getReadOperands(readOps);
    if (failed(subviews))
      return rewriter.notifyMatchFailure(
          contractOp, "Vector read op operands are not a subview");

    // Check the operation type MatMul, B-MatMul, or BR-MatMul
    SmallVector<vector::IteratorType> contractIteratorTypes =
        contractOp.getIteratorTypesArray();
    int reductionCount =
        std::count(contractIteratorTypes.begin(), contractIteratorTypes.end(),
                   vector::IteratorType::reduction);

    auto vectorReadOpLhsType = cast<ShapedType>(vectorReadOpLhs.getType());
    auto vectorReadOpRhsRank =
        (cast<ShapedType>(vectorReadOpRhs.getType())).getRank();

    if (reductionCount == 2 &&
        (vectorReadOpLhsType.getRank() != 3 || vectorReadOpRhsRank != 3))
      return rewriter.notifyMatchFailure(
          contractOp, "Invalid rank for batch reduce operation");

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          contractOp, "Batch matmul operation not supported yet");

    if (reductionCount > 2)
      return rewriter.notifyMatchFailure(
          contractOp, "The vector contract operation is not a gemm");

    // Check the K-dim to be 1
    int64_t K =
        vectorReadOpLhsType.getDimSize(vectorReadOpLhsType.getRank() - 1);
    if (K != 1)
      return rewriter.notifyMatchFailure(contractOp, "K dim is not 1");

    // Check whether the BR-matmul tiling + vector contract pattern matches for the
    // 4-nested loop structure
    auto loops = getNestedLoop(contractOp);
    if (failed(loops))
      return rewriter.notifyMatchFailure(
          contractOp, "Invalid loop nest in contract pattern");

    auto checkLoops = checkNestedLoop(*loops, *subviews);
    if (failed(checkLoops))
      return rewriter.notifyMatchFailure(
          contractOp, "Loops doesn't match the iv in subviews");

    auto nestedLoops = *loops;
    auto kForOp = nestedLoops[0];
    auto reductionForOp = nestedLoops[1];

    // Move the vector transfer read before the reduction and k loop
    rewriter.setInsertionPoint(reductionForOp);
    auto *cloneVectorReadOp = rewriter.clone(*vectorReadOpAcc);

    // Code to re-create the reduction and k loop with iter args
    auto vectorReadOpValue = cloneVectorReadOp->getResult(0);
    auto newReductionForOp = rewriter.create<scf::ForOp>(
        reductionForOp.getLoc(), reductionForOp.getLowerBound(),
        reductionForOp.getUpperBound(), reductionForOp.getStep(),
        ValueRange{vectorReadOpValue},
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp,
            Value ivNewReductionForOp, ValueRange iterArgsNewReductionForOp) {
          auto newKForOp = rewriter.create<scf::ForOp>(
              kForOp.getLoc(), kForOp.getLowerBound(), kForOp.getUpperBound(),
              kForOp.getStep(), iterArgsNewReductionForOp,
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                IRMapping mapper;
                mapper.map(reductionForOp.getInductionVar(),
                           ivNewReductionForOp);
                mapper.map(kForOp.getInductionVar(), ivNewKForOp);

                for (auto &op : kForOp.getBody()->without_terminator()) {
                  rewriterNewKForOp.clone(op, mapper);
                }
                rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp,
                                                       iterArgsNewKForOp);
              });
          rewriterNewReductionForOp.create<scf::YieldOp>(
              locNewReductionForOp, newKForOp.getResult(0));
        });

    // Code to hoist vector transfer write after reduction loop and also to
    // update the yield of k loop
    auto newKForOp =
        llvm::dyn_cast<scf::ForOp>(newReductionForOp.getBody()->front());
    Value newcontractOpValue;
    vector::TransferWriteOp vectorWriteOperation;
    Block *bodyBlock = newKForOp.getBody();
    for (auto &op : bodyBlock->getOperations()) {
      if (auto vectorContractOp = llvm::dyn_cast<vector::ContractionOp>(op)) {
        vectorContractOp.setOperand(vectorContractOp.getNumOperands() - 1,
                                    newKForOp.getRegionIterArgs()[0]);
        newcontractOpValue = vectorContractOp.getResult();
      }
      if (auto yieldOp = llvm::dyn_cast<scf::YieldOp>(op)) {
        yieldOp.setOperand(0, newcontractOpValue);
      }
      if (auto vectorWriteOp = llvm::dyn_cast<vector::TransferWriteOp>(op)) {
        vectorWriteOperation = vectorWriteOp;
      }
    }

    vectorWriteOperation.setOperand(0, newReductionForOp.getResult(0));
    vectorWriteOperation->moveBefore(reductionForOp);

    // Erase the old vector contract operation
    for (auto result : contractOp->getResults()) {
      for (auto *userOp : result.getUsers()) {
        userOp->erase();
      }
    }
    contractOp.erase();

    return success();
  }
};

void linalg::populateHoistVectorTransferPatterns(RewritePatternSet &patterns) {
  patterns.add<HoistVectorTransferOp>(patterns.getContext());
}
