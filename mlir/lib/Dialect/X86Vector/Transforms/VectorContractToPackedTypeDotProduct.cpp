//===- VectorContractToPackedTypeDotProduct.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
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

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

namespace {

// Returns true if the A or B matrix vector is packed (shuffled) to
// VNNI layout, already.
static bool isNonUnitDimOperandShuffled(Value nonUnitDimOperand) {
  if (Operation *defOp = nonUnitDimOperand.getDefiningOp()) {
    if (isa<vector::ShuffleOp>(defOp))
      return true;

    if (isa<vector::ShapeCastOp>(defOp)) {
      Operation *defOpShpCst = defOp->getOperand(0).getDefiningOp();
      if (isa<vector::ShuffleOp>(defOpShpCst))
        return true;
    }
  }

  return false;
}

static void rewriteUses(mlir::Value oldVal, mlir::Value newVal,
                        mlir::Operation *targetContract,
                        mlir::PatternRewriter &rewriter) {
  for (mlir::OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {

    mlir::Operation *user = use.getOwner();
    if (mlir::isa<mlir::vector::ContractionOp>(user) ||
        mlir::isa<mlir::scf::ForOp>(user)) {
      use.set(newVal);
    }
  }
}

// Function to convert the flat layout A or B matrix vector<32xbf16>
// into VNNI packed layout using the vpunpack operations
static void packNonUnitDimOperandToVNNI(mlir::PatternRewriter &rewriter,
                                        mlir::Operation *opA,
                                        mlir::Operation *opB,
                                        mlir::vector::ContractionOp contractA,
                                        mlir::vector::ContractionOp contractB,
                                        int64_t nonUnitDimAcc,
                                        mlir::VectorType Ty) {
  mlir::Operation *insertAfter = opA->isBeforeInBlock(opB) ? opB : opA;

  rewriter.setInsertionPointAfter(insertAfter);
  mlir::Location loc = insertAfter->getLoc();

  auto elemTy = Ty.getElementType();
  auto flatTy = mlir::VectorType::get(nonUnitDimAcc, elemTy);

  auto castA = mlir::vector::ShapeCastOp::create(rewriter, loc, flatTy,
                                                 opA->getResult(0));
  auto castB = mlir::vector::ShapeCastOp::create(rewriter, loc, flatTy,
                                                 opB->getResult(0));

  static constexpr int64_t maskLo[] = {
      0,  32, 1,  33, 2,  34, 3,  35, 8,  40, 9,  41, 10, 42, 11, 43,
      16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59};
  static constexpr int64_t maskHi[] = {
      4,  36, 5,  37, 6,  38, 7,  39, 12, 44, 13, 45, 14, 46, 15, 47,
      20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63};

  auto shuffleLo = mlir::vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                                   castB, maskLo);
  auto shuffleHi = mlir::vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                                   castB, maskHi);

  auto newA = mlir::vector::ShapeCastOp::create(rewriter, loc, Ty, shuffleLo);
  auto newB = mlir::vector::ShapeCastOp::create(rewriter, loc, Ty, shuffleHi);

  rewriteUses(opA->getResult(0), newA.getResult(), contractA, rewriter);
  rewriteUses(opB->getResult(0), newB.getResult(), contractB, rewriter);
}

// Implements packed type outer product contraction as a sequence
// of broadcast and packed dot-product operations.
//
// For example - for bf16 type (VNNI):
// ```
//   vector.contract <1x1x2xbf16>, <1x16x2xbf16> into <1x16xf32>
// ```
// to
// ```
//   vector.broadcast %lhs to <32xbf16>
//   x86vector.avx512.dot vector<32xbf16> -> vector<16xf32>
// ```
//
// For example - for bf16 type (Flat layout):
// ```
//   %1 = vector.load -> <2x16xbf16>
//   %2 = vector.load -> <2x16xbf16>
//   vector.contract <1x2xbf16>, %1 into <1x16xf32>
//   vector.contract <1x2xbf16>, %2 into <1x16xf32>
// ```
// to
// ```
//   %1 = vector.load -> <2x16xbf16>
//   %2 = vector.load -> <2x16xbf16>
//   %3 = vector.shuffle %1, %2 [0, 32, 1, ... 27, 59]
//   %4 = vector.shuffle %1, %2 [4, 36, 5, ... 31, 63]
//   vector.broadcast %lhs to <32xbf16>
//   x86vector.avx512.dot vector<32xbf16>, %3 -> vector<16xf32>
//   vector.broadcast %lhs to <32xbf16>
//   x86vector.avx512.dot vector<32xbf16>, %3 -> vector<16xf32>
// ```
struct VectorContractToPackedTypeDotProduct
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16() &&
        !lhsTy.getElementType().isSignlessInteger(8))
      return rewriter.notifyMatchFailure(
          contractOp, "Only BF16/Int8 lowering is supported.");

    unsigned int blockingFactor = lhsTy.getElementType().isBF16() ? 2 : 4;
    bool isVnni =
        isInVnniLayout(contractOp.getOperation(),
                       contractOp.getIndexingMapsArray(), blockingFactor);

    if (lhsTy.getElementType().isSignlessInteger(8) && !isVnni)
      return failure();

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accmulator type.");

    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimAcc;
    llvm::copy_if(accShape, std::back_inserter(nonUnitDimAcc),
                  [](int64_t dim) { return dim != 1; });
    if (nonUnitDimAcc.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "A or B should be a non-unit dim in acc.");

    int64_t nonUnitDimValue = nonUnitDimAcc.front();
    // Non-unit dimensions should match the vector length of BF16 or Int8
    // dot-product.
    if (lhsTy.getElementType().isBF16() && nonUnitDimValue != 4 &&
        nonUnitDimValue != 8 && nonUnitDimValue != 16)
      return rewriter.notifyMatchFailure(
          contractOp, "BF16 dot-product operation expects non-unit (LHR or "
                      "RHS) dim and acc dim of size 4/8/16.");

    if (lhsTy.getElementType().isSignlessInteger(8) && nonUnitDimValue != 4 &&
        nonUnitDimValue != 8 && nonUnitDimValue != 16 &&
        nonUnitDimAcc.front() == nonUnitDimValue)
      return rewriter.notifyMatchFailure(
          contractOp, "Int8 dot-product operation expects non-unit (LHR or "
                      "RHS) dim and acc dim of size 4/8/16.");

    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimLhs;
    llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                  [](int64_t dim) { return dim != 1; });

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimRhs;
    llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                  [](int64_t dim) { return dim != 1; });

    if ((nonUnitDimLhs.size() - 1) > 0 && (nonUnitDimRhs.size() - 1) > 0)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Excepts unit dimensions for either "
                                         "LHS or RHS shape.");

    if ((nonUnitDimLhs.size() - 1) != 1 && (nonUnitDimRhs.size() - 1) != 1)
      return rewriter.notifyMatchFailure(
          contractOp,
          "Excepts a one non-unit A/B dimension for either LHS or RHS shape.");

    bool rhsHasMultipleNonUnitDims = (nonUnitDimRhs.size() - 1) > 0;
    int64_t extraFlatDim = rhsHasMultipleNonUnitDims ? nonUnitDimLhs.front()
                                                     : nonUnitDimRhs.front();

    if (!isVnni && (extraFlatDim != blockingFactor))
      return rewriter.notifyMatchFailure(
          contractOp, "The K or reduction dim for flat layout should be 2.");

    if ((lhsTy.getElementType().isBF16() && !accTy.getElementType().isF32()) ||
        (lhsTy.getElementType().isSignlessInteger(8) &&
         !accTy.getElementType().isSignlessInteger(32)))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only F32 for BF16 or Int32 for Int8 "
                                         "accumulation type is supported.");

    Value unitDimOperand =
        rhsHasMultipleNonUnitDims ? contractOp.getLhs() : contractOp.getRhs();
    Value nonUnitDimOperand =
        rhsHasMultipleNonUnitDims ? contractOp.getRhs() : contractOp.getLhs();

    // If the A or B matrix vector of the contact operation is not packed, then
    // find it's pair contract operation and pack (shuffle) them to VNNI packed.
    if (!isVnni) {
      vector::ContractionOp pairContractOp;
      Operation *nextOp = contractOp;
      while ((nextOp = nextOp->getNextNode())) {
        auto contOp = dyn_cast<vector::ContractionOp>(nextOp);

        if (!contOp)
          continue;

        if (validatePairVectorContract(contractOp, contOp,
                                       rhsHasMultipleNonUnitDims,
                                       nonUnitDimValue)) {
          pairContractOp = contOp;
          break;
        }
      }

      // If the accumulators are shuffled we get nullptr else the
      // transfer_read or load operations.
      Operation *accRead =
          traceToVectorReadLikeParentOperation(contractOp.getAcc());

      if (!pairContractOp &&
          (!isNonUnitDimOperandShuffled(nonUnitDimOperand) || accRead))
        return rewriter.notifyMatchFailure(contractOp,
                                           "Could not find a contract pair");

      // Validate and shuffle the accumulator
      if (accRead) {
        // Trace back to the load or transfer_read operations of the contract
        // accumulators.
        Operation *accReadOp0 =
            traceToVectorReadLikeParentOperation(contractOp.getAcc());
        Operation *accReadOp1 =
            traceToVectorReadLikeParentOperation(pairContractOp.getAcc());

        // Iterate down to find the users of contact operations until it is
        // store or transfer_write.
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
              contractOp,
              "The use of contract operations are neither vector.store "
              "or transfer_write or has multiple users.");

        if (contractOp->getBlock() == accReadOp1->getBlock() &&
            contractOp->isBeforeInBlock(accReadOp1))
          return rewriter.notifyMatchFailure(
              contractOp,
              "The load/read operation of pair contract operation is "
              "after the contractOp");

        if (pairContractOp->getBlock() == resultWriteOp0->getBlock() &&
            resultWriteOp0->isBeforeInBlock(pairContractOp))
          return rewriter.notifyMatchFailure(
              contractOp, "The store/write operation of contract operation is "
                          "before the pair contract operation");
        // Shuffle the accumulators of the contract operations.
        LogicalResult readShuffle =
            shuffleAfterReadLikeOp(rewriter, accReadOp0, accReadOp1, contractOp,
                                   pairContractOp, nonUnitDimValue, accTy);

        if (failed(readShuffle))
          return rewriter.notifyMatchFailure(
              contractOp, "Accumulator read is not by transfer_read or load");

        // Shuffle the output of contract operations before it's use.
        LogicalResult writeShuffle = shuffleBeforeWriteLikeOp(
            rewriter, resultWriteOp0, resultWriteOp1, nonUnitDimValue, accTy);

        if (failed(writeShuffle))
          return rewriter.notifyMatchFailure(
              contractOp,
              "Write to accumulator is not by transfer_write or store");
      }

      if (!isNonUnitDimOperandShuffled(nonUnitDimOperand)) {
        Value nonUnitDimOperandPairContract = rhsHasMultipleNonUnitDims
                                                  ? pairContractOp.getRhs()
                                                  : pairContractOp.getLhs();

        // Get the non-packed A or B matrix's vector<32xbf16> elements.
        Operation *nonUnitDimReadOp =
            traceToVectorReadLikeParentOperation(nonUnitDimOperand);
        Operation *nonUnitDimReadOpPairContract =
            traceToVectorReadLikeParentOperation(nonUnitDimOperandPairContract);

        if (!nonUnitDimReadOp || !nonUnitDimReadOpPairContract)
          return rewriter.notifyMatchFailure(
              contractOp, "Could not find a valid contract pair");

        if (contractOp->getBlock() ==
                nonUnitDimReadOpPairContract->getBlock() &&
            contractOp->isBeforeInBlock(nonUnitDimReadOpPairContract))
          return rewriter.notifyMatchFailure(
              contractOp,
              "The load/read operation of pair contract operation is "
              "after the contractOp");

        VectorType nonUnitDimTy = rhsHasMultipleNonUnitDims
                                      ? contractOp.getRhsType()
                                      : contractOp.getLhsType();

        packNonUnitDimOperandToVNNI(
            rewriter, nonUnitDimReadOp, nonUnitDimReadOpPairContract,
            contractOp, pairContractOp, blockingFactor * nonUnitDimValue,
            nonUnitDimTy);

        nonUnitDimOperand = rhsHasMultipleNonUnitDims ? contractOp.getRhs()
                                                      : contractOp.getLhs();
      }
    }

    rewriter.setInsertionPoint(contractOp);
    auto loc = contractOp.getLoc();
    auto castAcc = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(nonUnitDimAcc.front(), accTy.getElementType()),
        contractOp.getAcc());

    VectorType nonUnitDimTy = rhsHasMultipleNonUnitDims
                                  ? contractOp.getRhsType()
                                  : contractOp.getLhsType();
    VectorType unitDimTy = rhsHasMultipleNonUnitDims ? contractOp.getLhsType()
                                                     : contractOp.getRhsType();

    Value dp;

    auto castNonUnitDim = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(blockingFactor * nonUnitDimValue,
                        nonUnitDimTy.getElementType()),
        nonUnitDimOperand);

    auto castUnitDim = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(blockingFactor, unitDimTy.getElementType()),
        unitDimOperand);
    auto bitcastUnitDim = vector::BitCastOp::create(
        rewriter, loc, VectorType::get({1}, rewriter.getIntegerType(32)),
        castUnitDim);
    auto broadcastUnitDim = vector::BroadcastOp::create(
        rewriter, loc,
        VectorType::get({nonUnitDimValue}, rewriter.getIntegerType(32)),
        bitcastUnitDim);
    auto bitcastUnitDimPkType = vector::BitCastOp::create(
        rewriter, loc, castNonUnitDim.getResult().getType(), broadcastUnitDim);

    if (lhsTy.getElementType().isBF16()) {
      dp = x86vector::DotBF16Op::create(
          rewriter, loc,
          VectorType::get(nonUnitDimValue, rewriter.getF32Type()), castAcc,
          bitcastUnitDimPkType, castNonUnitDim);
    }

    if (lhsTy.getElementType().isSignlessInteger(8)) {
      if (nonUnitDimAcc.front() == 16) {
        dp = x86vector::AVX10DotInt8Op::create(
            rewriter, loc,
            VectorType::get(nonUnitDimValue, rewriter.getIntegerType(32)),
            castAcc, bitcastUnitDimPkType, castNonUnitDim);
      } else {
        dp = x86vector::DotInt8Op::create(
            rewriter, loc,
            VectorType::get(nonUnitDimValue, rewriter.getIntegerType(32)),
            castAcc, bitcastUnitDimPkType, castNonUnitDim);
      }
    }

    if (!dp)
      return failure();

    auto castDp = vector::ShapeCastOp::create(rewriter, loc, accTy, dp);
    rewriter.replaceOp(contractOp, castDp);
    return success();
  }
};

} // namespace

void x86vector::populateVectorContractToPackedTypeDotProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToPackedTypeDotProduct>(patterns.getContext());
}
