//===- VectorContractBF16ToFMA.cpp-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Verifies that the LHS and RHS operands of a vector.contract are load or
// vector.transfer_read operations on a memref source buffer, and checks
// their bounds, dimensions, offsets, and strides.
static bool validateVectorContractOperands(Value prodOp, bool isVnni) {
  Operation *defOp = prodOp.getDefiningOp();
  if (!defOp)
    return false;

  if (auto readOp = prodOp.getDefiningOp<mlir::vector::TransferReadOp>()) {
    if (readOp.hasOutOfBoundsDim())
      return false;

    if (!readOp.getPermutationMap().isMinorIdentity())
      return false;
  }

  Value srcBuff;
  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<Operation *>(defOp).Case<TransferReadOp, LoadOp>(
      [&](auto readOp) {
        srcBuff = readOp.getOperand(0);
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
      });

  if (!srcBuff)
    return false;

  // Return false, if the source is not a memref type
  Type srcType = srcBuff.getType();
  if (!llvm::isa<MemRefType>(srcType))
    return false;

  // Return false if the two innermost strides of the memref are not contiguous.
  // The x86vector.avx.cvt.packed.even/odd.indexed_to_f32 operations require
  // an eight-element tuple of bf16 values to be contiguous.
  int dimsToCheck = isVnni ? 2 : 1;
  if (!cast<mlir::MemRefType>(srcType).areTrailingDimsContiguous(dimsToCheck))
    return false;

  // Return false if the vnni offset of load or transfer_read is not zero.
  if (isVnni && getConstantIntValue(indexVals.back()) != 0)
    return false;

  return true;
}

// This function retrieves the source operation of the load or transfer
// reads and creates subviews for the BF16 packed-operations to
// broadcast or load BF16 elements as F32 packed elements.
//
// Example(1) Unit Dim:
// ```
//   vector.load %arg0[%c0, %c0, %c0]:memref<4x1x2xbf16>,vector<1x1x2xbf16>
// ```
// to
// ```
//   memref.subview %arg0[%c0,%c0,%c1]:memref<4x1x2xbf16> to memref<1x1x1xbf16>
//   memref.subview %arg0[%c0,%c0,%c0]:memref<4x1x2xbf16> to memref<1x1x1xbf16>
// ```
//
// Example(2) Non-unit Dim:
// ```
//   vector.load %arg1[%c0, %c0, %c0]:memref<1x32x2xbf16>,vector<1x8x2xbf16>
// ```
// to
// ```
//   memref.subview %arg1[%c0,%c0,%c0]:memref<1x32x2xbf16> to memref<1x8x2xbf16>
// ```
static SmallVector<memref::SubViewOp>
getSubviewFromVectorInput(Location loc, PatternRewriter &rewriter, Value prodOp,
                          ArrayRef<int64_t> nonUnitDimShape, bool isUnitDim,
                          bool isVNNI) {

  Operation *defOp = prodOp.getDefiningOp();

  Value srcBuff;
  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<Operation *>(defOp).Case<TransferReadOp, LoadOp>(
      [&](auto readOp) {
        srcBuff = readOp.getOperand(0);
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
      });

  int64_t mnDimSize = 1;
  unsigned mnDimIdx = 0;

  if (!isUnitDim) {
    for (auto it : llvm::enumerate(nonUnitDimShape)) {
      if (it.value() != 1) {
        mnDimSize = it.value();
        mnDimIdx = it.index();
        break;
      }
    }
  }

  auto one = rewriter.getIndexAttr(1);
  SmallVector<memref::SubViewOp> subviews;

  if (!isVNNI) {
    SmallVector<OpFoldResult> strides(indexVals.size(), one);
    SmallVector<OpFoldResult> sizes(indexVals.size(), one);
    // Retrive twice the nonUnit dim BF16 element for both even and odd
    // index elements.
    if (!isUnitDim)
      mnDimSize = 2 * mnDimSize;
    sizes[mnDimIdx] = rewriter.getIndexAttr(mnDimSize);
    auto subview = memref::SubViewOp::create(rewriter, loc, srcBuff, indexVals,
                                             sizes, strides);
    subviews.push_back(subview);
    return subviews;
  }

  int vnniDimSize = isUnitDim ? 1 : 2;
  auto nonVNNIDimSize = indexVals.size() - 1;
  // Create the size and stride offsets.
  SmallVector<OpFoldResult> strides(indexVals.size(), one);
  SmallVector<OpFoldResult> sizes(nonVNNIDimSize, one);

  sizes.push_back(rewriter.getIndexAttr(vnniDimSize));

  // update the unit/nonUnit Dim size either it is A(LHS) or B(RHS).
  sizes[mnDimIdx] = rewriter.getIndexAttr(mnDimSize);

  // for unitDim, first broadcast odd element, so index is set to 1.
  if (isUnitDim)
    indexVals[indexVals.size() - 1] = rewriter.getIndexAttr(1);

  auto subview = memref::SubViewOp::create(rewriter, loc, srcBuff, indexVals,
                                           sizes, strides);
  subviews.push_back(subview);

  // For unit-dims, two subviews should be created for the odd and even
  // element in the VNNI tuple (2xbf16) because x86vector.avx.bcst_to_f32.packed
  // op loads and broadcast the first BF16 element into packed F32. It
  // cannot distinguish between even and odd BF16 elements within a
  // packed pair.
  //
  // Example:
  // memref.subview %arg0[%c0,%c1]:memref<1x2xbf16> to memref<1x1xbf16> // Odd
  // memref.subview %arg0[%c0,%c0]:memref<1x2xbf16> to memref<1x1xbf16> // Even
  if (mnDimSize == 1) {
    indexVals[indexVals.size() - 1] = rewriter.getIndexAttr(0);
    sizes[indexVals.size() - 1] = rewriter.getIndexAttr(1);

    auto unitDimEvenIdxSubview = memref::SubViewOp::create(
        rewriter, loc, srcBuff, indexVals, sizes, strides);
    subviews.push_back(unitDimEvenIdxSubview);
  }

  return subviews;
}

// Implements outer product contraction as a sequence of BF16-packed
// operation even/odd loads and FMA operations.
//
// For example (VNNI packed):
// ```
//   %1 = vector.load from memref (%m1) -> vector<1x1x2xbf16>
//   %2 = vector.load from memref (%m2) -> vector<1x8x2xbf16>
//   return vector.contract %1, %2, %arg1
// ```
// to
// ```
//   %1 = x86vector.avx.bcst_to_f32.packed %m1[c1] -> vector<8xf32>
//   %2 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %m2 -> vector<8xf32>
//   %3 = vector.fma %1, %2, %arg1
//   %4 = x86vector.avx.bcst_to_f32.packed %m1[c0] -> vector<8xf32>
//   %5 = x86vector.avx.cvt.packed.even.indexed_to_f32 %m2 -> vector<8xf32>
//   return vector.fma %4, %5, %3
// ```
//
// For example (Flat layout):
// ```
//   %1 = vector.load from memref (%m1) -> vector<1x1xbf16>
//   %2 = vector.load from memref (%m2) -> vector<1x8xbf16>
//   %3 = vector.contract %1, %2, %arg1
//   %4 = vector.load from memref (%m2) -> vector<1x8xbf16>
//   %5 = vector.contract %1, %4, %arg2
//   scf.yield %3, %4
// ```
// to
// ```
//   %1 = x86vector.avx.bcst_to_f32.packed %m1[c0] -> vector<8xf32>
//   %2 = x86vector.avx.cvt.packed.even.indexed_to_f32 %m2 -> vector<8xf32>
//   %3 = vector.fma %1, %2, %arg1
//   %4 = x86vector.avx.cvt.packed.odd.indexed_to_f32 %m2 -> vector<8xf32>
//   %5 = vector.fma %1, %4, %arg2
//   scf.yield %3, %5
struct VectorContractBF16ToFMA
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

    bool isVnni = isInVnniLayout(contractOp.getOperation(),
                                 contractOp.getIndexingMapsArray(),
                                 /*blockingFactor=*/2);

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

    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimLhs;
    llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                  [](int64_t dim) { return dim != 1; });

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimRhs;
    llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                  [](int64_t dim) { return dim != 1; });

    if (isVnni && (nonUnitDimLhs.size() - 1) > 0 &&
        (nonUnitDimRhs.size() - 1) > 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Excepts unit dimensions for either "
                                         "LHS or RHS shape other than VNNI.");

    if (isVnni && (nonUnitDimLhs.size() - 1) != 1 &&
        (nonUnitDimRhs.size() - 1) != 1)
      return rewriter.notifyMatchFailure(
          contractOp,
          "Excepts a one non-unit A/B dimension for either LHS or RHS shape.");

    if (!isVnni && nonUnitDimLhs.size() > 0 && nonUnitDimRhs.size() > 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Excepts unit dimensions for either "
                                         "LHS or RHS shape.");

    if (!isVnni && nonUnitDimLhs.size() != 1 && nonUnitDimRhs.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp,
          "Excepts a one non-unit A/B dimension for either LHS or RHS shape.");

    // Non-unit dimensions should match the vector length of BF16.
    unsigned int nonUnitDim = nonUnitDimAcc.front();
    if (nonUnitDim != 4 && nonUnitDim != 8)
      return rewriter.notifyMatchFailure(
          contractOp, "BF16 packed load operation expects non-unit (LHR or "
                      "RHS) dim and acc dim of size 4/8.");

    if (!validateVectorContractOperands(contractOp.getLhs(), isVnni) ||
        !validateVectorContractOperands(contractOp.getRhs(), isVnni)) {
      return rewriter.notifyMatchFailure(
          contractOp, "The LHS or RHS is in an invalid format. Either it has "
                      "false in-bounds, "
                      "a non-identity permutation map, a non-zero VNNI offset, "
                      "a non-memref "
                      "source, or a non-unit VNNI stride");
    }

    // Lower vector.contract to FMAs with help of BF16 packed ops.
    auto loc = contractOp.getLoc();

    // create the unit-dimension LHS or RHS subview and the
    // corresponding non-unit dimension LHS or RHS subview on the other-side.
    // For example, if LHS has type vector<1x1x2xbf16> and RHS has type
    // vector<1x8x2xbf16>, we create two subview for the LHS and one subview
    // for the RHS. In the opposite case (non-unit dimension on the LHS), we
    // do vice-versa.

    bool rhsHasMultipleNonUnitDims = nonUnitDimRhs.size() > 0;
    if (isVnni) {
      rhsHasMultipleNonUnitDims = (nonUnitDimRhs.size() - 1) > 0;
    }

    // Select which operand is "unit" and which is "non-unit".
    Value unitSrc =
        rhsHasMultipleNonUnitDims ? contractOp.getLhs() : contractOp.getRhs();
    Value nonUnitSrc =
        rhsHasMultipleNonUnitDims ? contractOp.getRhs() : contractOp.getLhs();

    ArrayRef<int64_t> nonUnitDimShape =
        rhsHasMultipleNonUnitDims ? rhsShape : lhsShape;

    // Get the pair vector.contract operation. The pair is decided on:
    //  (1) - the unitDim operand Lhs or Rhs should be same,
    //  (2) - the defining source memref should be same for nonUnitDim
    //  operation, (3) - the nonUnit dim offset difference between the
    //  vector.contracts should be 8.
    vector::ContractionOp pairContractOp;
    if (!isVnni) {
      Operation *nextOp = contractOp;
      while ((nextOp = nextOp->getNextNode())) {
        auto contOp = dyn_cast<vector::ContractionOp>(nextOp);

        if (!contOp)
          continue;

        if (validatePairVectorContract(contractOp, contOp,
                                       rhsHasMultipleNonUnitDims,
                                       nonUnitDimAcc.front())) {
          pairContractOp = contOp;
          break;
        }
      }

      if (!pairContractOp)
        return failure();

      Operation *accReadOp0 =
          traceToVectorReadLikeParentOperation(contractOp.getAcc());
      Operation *accReadOp1 =
          traceToVectorReadLikeParentOperation(pairContractOp.getAcc());

      // Iterate down to find the users of contact operations until it is store
      // or transfer_write.
      Operation *resultWriteOp0 =
          traceToVectorWriteLikeUserOperation(contractOp.getResult());
      Operation *resultWriteOp1 =
          traceToVectorWriteLikeUserOperation(pairContractOp.getResult());

      if (!accReadOp0 || !accReadOp1)
        return rewriter.notifyMatchFailure(
            contractOp,
            "Operand doesn't have load or transfer_read as its parent op");

      if (!resultWriteOp0 || !resultWriteOp1)
        return rewriter.notifyMatchFailure(
            contractOp,
            "The use of contract operations are neither vector.store "
            "or transfer_write or has multiple users");

      if (contractOp->getBlock() == accReadOp1->getBlock() &&
          contractOp->isBeforeInBlock(accReadOp1))
        return rewriter.notifyMatchFailure(
            contractOp, "The load/read operation of pair contract operation is "
                        "after the contractOp");

      if (pairContractOp->getBlock() == resultWriteOp0->getBlock() &&
          resultWriteOp0->isBeforeInBlock(pairContractOp)) {
        return rewriter.notifyMatchFailure(
            contractOp, "The store/write operation of contract operation is "
                        "before the pair contract operation");
      }
    }

    // Build subviews.
    auto unitDimSubview = getSubviewFromVectorInput(
        loc, rewriter, unitSrc, nonUnitDimShape, true, isVnni);

    auto nonUnitDimSubview = getSubviewFromVectorInput(
        loc, rewriter, nonUnitSrc, nonUnitDimShape, false, isVnni);

    auto castAcc = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(nonUnitDimAcc.front(), accTy.getElementType()),
        contractOp.getAcc());
    VectorType dstType =
        VectorType::get(nonUnitDimAcc.front(), rewriter.getF32Type());

    if (!isVnni) {

      // Validate and shuffle the accumulator
      Operation *accReadOp0 =
          traceToVectorReadLikeParentOperation(contractOp.getAcc());
      Operation *accReadOp1 =
          traceToVectorReadLikeParentOperation(pairContractOp.getAcc());

      // Iterate down to find the users of contact operations until it is store
      // or transfer_write.
      Operation *resultWriteOp0 =
          traceToVectorWriteLikeUserOperation(contractOp.getResult());
      Operation *resultWriteOp1 =
          traceToVectorWriteLikeUserOperation(pairContractOp.getResult());

      // Shuffle the accumulators of the contract operations.
      LogicalResult readShuffle =
          shuffleAfterReadLikeOp(rewriter, accReadOp0, accReadOp1, contractOp,
                                 pairContractOp, nonUnitDim, accTy);

      if (failed(readShuffle))
        return rewriter.notifyMatchFailure(
            contractOp, "Accumulator read is not by transfer_read or load");

      // Shuffle the output of contract operations before its use.
      LogicalResult writeShuffle = shuffleBeforeWriteLikeOp(
          rewriter, resultWriteOp0, resultWriteOp1, nonUnitDim, accTy);

      if (failed(writeShuffle))
        return rewriter.notifyMatchFailure(
            contractOp,
            "Write to accumulator is not by transfer_write or store");

      rewriter.setInsertionPoint(contractOp);
      castAcc = vector::ShapeCastOp::create(
          rewriter, loc,
          VectorType::get(nonUnitDimAcc.front(), accTy.getElementType()),
          contractOp.getAcc());

      auto loadBcstBF16ElementToF32 = x86vector::BcstToPackedF32Op::create(
          rewriter, loc, dstType, unitDimSubview[0]);
      auto loadEvenIdxElementF32 =
          x86vector::CvtPackedEvenIndexedToF32Op::create(rewriter, loc, dstType,
                                                         nonUnitDimSubview[0]);
      auto evenIdxFMA =
          vector::FMAOp::create(rewriter, loc, loadBcstBF16ElementToF32,
                                loadEvenIdxElementF32, castAcc);
      auto castEvenFma =
          vector::ShapeCastOp::create(rewriter, loc, accTy, evenIdxFMA);
      rewriter.replaceOp(contractOp, castEvenFma);

      rewriter.setInsertionPoint(pairContractOp);
      auto pairContOpLoc = pairContractOp.getLoc();
      VectorType accTyPairCont =
          dyn_cast<VectorType>(pairContractOp.getAccType());
      auto castAccPairCont = vector::ShapeCastOp::create(
          rewriter, pairContOpLoc,
          VectorType::get(nonUnitDimAcc.front(),
                          accTyPairCont.getElementType()),
          pairContractOp.getAcc());

      auto loadOddIdxElementF32 = x86vector::CvtPackedOddIndexedToF32Op::create(
          rewriter, pairContOpLoc, dstType, nonUnitDimSubview[0]);
      auto oddIdxFMA = vector::FMAOp::create(
          rewriter, pairContOpLoc, loadBcstBF16ElementToF32,
          loadOddIdxElementF32, castAccPairCont);
      auto castOddFma = vector::ShapeCastOp::create(rewriter, pairContOpLoc,
                                                    accTyPairCont, oddIdxFMA);
      rewriter.replaceOp(pairContractOp, castOddFma);

      return success();
    }

    // Load, broadcast, and do FMA for odd indexed BF16 elements.
    auto loadBcstOddIdxElementToF32 = x86vector::BcstToPackedF32Op::create(
        rewriter, loc, dstType, unitDimSubview[0]);
    auto loadOddIdxElementF32 = x86vector::CvtPackedOddIndexedToF32Op::create(
        rewriter, loc, dstType, nonUnitDimSubview[0]);
    auto oddIdxFMA =
        vector::FMAOp::create(rewriter, loc, loadBcstOddIdxElementToF32,
                              loadOddIdxElementF32, castAcc);

    // Load, broadcast, and do FMA for even indexed BF16 elements.
    auto loadBcstEvenIdxElementToF32 = x86vector::BcstToPackedF32Op::create(
        rewriter, loc, dstType, unitDimSubview[1]);
    auto loadEvenIdxElementF32 = x86vector::CvtPackedEvenIndexedToF32Op::create(
        rewriter, loc, dstType, nonUnitDimSubview[0]);
    vector::FMAOp fma =
        vector::FMAOp::create(rewriter, loc, loadBcstEvenIdxElementToF32,
                              loadEvenIdxElementF32, oddIdxFMA);

    auto castFma = vector::ShapeCastOp::create(rewriter, loc, accTy, fma);
    rewriter.replaceOp(contractOp, castFma);
    return success();
  }
};

void x86vector::populateVectorContractBF16ToFMAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractBF16ToFMA>(patterns.getContext());
}
