//===- ShuffleBF16VectorContractResult.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/Utils/X86VectorUtils.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

// Shuffle the output of BF16 type flat layout vector.contract operations
//
// For example:
// ```
//   %1 = vector.load -> vector<1x1xbf16>
//   %2 = vector.load from memref (%m1) -> vector<1x8xbf16>
//   %3 = vector.load from memref (%m1) -> vector<1x8xbf16>
//   %4 = vector.contract %1, %2, %arg0 ->  vector<1x8xf32>
//   %5 = vector.contract %1, %3, %arg1 ->  vector<1x8xf32>
//   vector.store %4, %m1
//   vector.store %5, %m1
// ```
// to
// ```
//   %1 = vector.load -> vector<1x1xbf16>
//   %2 = vector.load from memref (%m1) -> vector<1x8xbf16>
//   %3 = vector.load from memref (%m1) -> vector<1x8xbf16>
//   %4 = vector.shuffle %arg0, %arg1 [0, 8, 1, 9, 2, 10, 3, 11]
//   %5 = vector.shuffle %arg0, %arg1 [4, 12, 5, 13, 6, 14, 7, 15]
//   %6 = vector.contract %1, %2, %4 ->  vector<1x8xf32>
//   %7 = vector.contract %1, %3, %5 ->  vector<1x8xf32>
//   %8 = vector.shuffle %6, %7 [0, 8, 1, 9, 2, 10, 3, 11]
//   %9 = vector.shuffle %6, %7 [4, 12, 5, 13, 6, 14, 7, 15]
//   vector.store %8, %m1
//   vector.store %9, %m1
//```
struct ShuffleBF16VectorContractResult
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    // TODO: Move this validation to a common utility folder. Planned to
    // do once (code refactoring), all architecture specific nanokernel
    // passes are merged into the repo.
    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only BF16 lowering is supported.");

    if (isInVnniLayout(contractOp.getOperation(),
                       contractOp.getIndexingMapsArray(),
                       /*blockingFactor=*/2))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Input matrices in VNNI format.");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accmulator type.");

    if (!accTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          contractOp, "Only F32 acumulation supported for BF16 type.");

    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimAcc;
    llvm::copy_if(accShape, std::back_inserter(nonUnitDimAcc),
                  [](int64_t dim) { return dim != 1; });

    if (nonUnitDimAcc.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "A or B should be a non-unit dim in acc.");

    int64_t nonUnitDimValue = nonUnitDimAcc.front();

    if (nonUnitDimValue != 8 && nonUnitDimValue != 16)
      return rewriter.notifyMatchFailure(
          contractOp, "The accumulator dimension should be 8 or 16");

    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimLhs;
    llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                  [](int64_t dim) { return dim != 1; });

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimRhs;
    llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                  [](int64_t dim) { return dim != 1; });

    vector::ContractionOp pairContractOp;
    bool rhsHasMultipleNonUnitDims =
        nonUnitDimRhs.size() > nonUnitDimLhs.size();

    // Get the pair vector.contract operation. The pair is decided on:
    //  (1) - the unitDim operand Lhs or Rhs should be same,
    //  (2) - the defining source memref should be same for nonUnitDim
    //  operation, (3) - the nonUnit dim offset difference between the
    //  vector.contracts should be 8.
    Operation *nextOp = contractOp;
    while ((nextOp = nextOp->getNextNode())) {
      auto contOp = dyn_cast<vector::ContractionOp>(nextOp);

      if (!contOp)
        continue;

      if (validatePairVectorContract(
              contractOp, contOp, rhsHasMultipleNonUnitDims, nonUnitDimValue)) {
        pairContractOp = contOp;
        break;
      }
    }

    if (!pairContractOp)
      return rewriter.notifyMatchFailure(
          contractOp, "Coudn't find pair contract operation for shuffling");

    // Trace back to the load or transfer_read operations of the contract
    // accumulators.
    Operation *accReadOp0 =
        traceToVectorReadLikeParentOperation(contractOp.getAcc());
    Operation *accReadOp1 =
        traceToVectorReadLikeParentOperation(pairContractOp.getAcc());

    // Iterate dowm to find the users of contact operations until it is store or
    // transfer_write.
    Operation *resultWriteOp0 =
        traceToVectorWriteLikeUserOperation(contractOp.getResult());
    Operation *resultWriteOp1 =
        traceToVectorWriteLikeUserOperation(pairContractOp.getResult());

    if (!accReadOp0 || !accReadOp1)
      return rewriter.notifyMatchFailure(
          contractOp,
          "Operands doesn't have load or transfer_read as it's parent op");

    if (!resultWriteOp0 || !resultWriteOp1)
      return rewriter.notifyMatchFailure(
          contractOp, "The use of contract operations are neither vector.store "
                      "or transfer_write");

    if (contractOp->getBlock() == accReadOp1->getBlock() &&
        contractOp->isBeforeInBlock(accReadOp1))
      return rewriter.notifyMatchFailure(
          contractOp, "The load/read operation of pair contract operation is "
                      "after the contractOp");

    if (pairContractOp->getBlock() == resultWriteOp0->getBlock() &&
        resultWriteOp0->isBeforeInBlock(pairContractOp))
      return rewriter.notifyMatchFailure(
          contractOp, "The store/write operation of contract operation is "
                      "before the pair contract operation");

    // Shuffle the accumulators of the contract operations.
    shuffleAfterReadLikeOp(rewriter, accReadOp0, accReadOp1, contractOp,
                           pairContractOp, nonUnitDimValue, accTy);

    // Shuffle the output of contract operations before it's use.
    shuffleBeforeWriteLikeOp(rewriter, resultWriteOp0, resultWriteOp1,
                             nonUnitDimValue, accTy);

    return success();
  }
};

void x86vector::populateShuffleBF16VectorContractResultPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ShuffleBF16VectorContractResult>(patterns.getContext());
}
