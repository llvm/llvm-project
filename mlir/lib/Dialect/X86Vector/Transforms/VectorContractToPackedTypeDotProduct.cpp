//===- VectorContractToPackedTypeDotProduct.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

static FailureOr<SmallVector<mlir::utils::IteratorType>>
inferIteratorsFromOutMap(AffineMap map) {
  if (!map.isProjectedPermutation())
    return failure();
  SmallVector<mlir::utils::IteratorType> iterators(
      map.getNumDims(), mlir::utils::IteratorType::reduction);
  for (auto expr : map.getResults())
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      iterators[dim.getPosition()] = mlir::utils::IteratorType::parallel;
  return iterators;
}

static bool isInVnniLayout(Operation *op, ArrayRef<AffineMap> indexingMaps,
                           std::optional<unsigned> blockingFactor) {
  // Narrow down type operations - VNNI only applies to contractions.
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(indexingMaps);
  if (failed(dims))
    return false;

  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto typeA = dyn_cast<ShapedType>(matA.getType());
  auto typeB = dyn_cast<ShapedType>(matB.getType());
  unsigned rankA = typeA.getRank();
  unsigned rankB = typeB.getRank();
  // VNNI format requires at least 1 parallel and 2 reduction dimensions.
  if (rankA < 3 || rankB < 3)
    return false;

  // At least two reduction dimensions are expected:
  // one for the VNNI factor and one for the K dimension
  if (dims->k.size() < 2)
    return false;

  // Validate affine maps - VNNI computation should be defined by the two
  // innermost reduction iterators.
  // The input matrix dimensions layout must match the following:
  //   - matrix A - [...][K/vnniFactor][vnniFactor]
  //   - matrix B - [...][K/vnniFactor][N][vnniFactor]
  auto maybeIters = inferIteratorsFromOutMap(indexingMaps[2]);
  if (failed(maybeIters))
    return false;
  SmallVector<mlir::utils::IteratorType> iteratorTypes = *maybeIters;
  AffineMap mapA = indexingMaps[0];
  AffineMap mapB = indexingMaps[1];

  auto vnniDimA = dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1));
  auto vnniDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 1));
  if (!vnniDimA || !vnniDimB || vnniDimA != vnniDimB ||
      iteratorTypes[vnniDimA.getPosition()] !=
          mlir::utils::IteratorType::reduction)
    return false;
  auto redDimA = dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 2));
  auto redDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 3));
  if (!redDimA || !redDimB || redDimA != redDimB ||
      iteratorTypes[redDimA.getPosition()] !=
          mlir::utils::IteratorType::reduction)
    return false;
  auto parallelDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 2));
  if (!parallelDimB || iteratorTypes[parallelDimB.getPosition()] !=
                           mlir::utils::IteratorType::parallel)
    return false;

  // VNNI factor must be:
  //   - the innermost inputs' dimension
  //   - statically known
  //   - multiple of 2 or equal to the specified factor
  auto vnniDimSize = typeB.getShape().back();
  if (vnniDimSize == ShapedType::kDynamic || vnniDimSize == 0 ||
      vnniDimSize % 2 != 0)
    return false;
  if (typeA.getShape().back() != vnniDimSize)
    return false;
  if (blockingFactor && vnniDimSize != *blockingFactor)
    return false;

  // The split reduction dimension size should also match.
  if (typeA.getShape().end()[-2] != typeB.getShape().end()[-3])
    return false;

  return true;
}

// Implements packed type outer product contraction as a sequence
// of broadcast and packed dot-product operations.
//
// For example - for F32 type:
// ```
//   vector.contract <1x1x2xbf16>, <1x16x2xbf16> into <1x16xf32>
// ```
// to
// ```
//   vector.broadcast %lhs to <32xbf16>
//   x86vector.avx512.dot vector<32xbf16> -> vector<16xf32>
// ```
struct VectorContractToPackedTypeDotProduct
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind");

    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16() &&
        !lhsTy.getElementType().isSignlessInteger(8))
      return rewriter.notifyMatchFailure(
          contractOp, "Only BF16/Int8 lowering is supported.");

    if (lhsTy.getElementType().isBF16() &&
        !isInVnniLayout(contractOp.getOperation(),
                        contractOp.getIndexingMapsArray(), 2))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Input matrices not in VNNI format");

    if (lhsTy.getElementType().isSignlessInteger(8) &&
        !isInVnniLayout(contractOp.getOperation(),
                        contractOp.getIndexingMapsArray(), 4))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Input matrices not in VNNI format");

    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    llvm::SmallVector<int64_t> dimsLhs;
    llvm::copy_if(lhsShape, std::back_inserter(dimsLhs),
                  [](int64_t dim) { return dim != 1; });

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> dimsRhs;
    llvm::copy_if(rhsShape, std::back_inserter(dimsRhs),
                  [](int64_t dim) { return dim != 1; });

    if ((dimsLhs.size() - 1) > 0 && (dimsRhs.size() - 1) > 0)
      return rewriter.notifyMatchFailure(
          contractOp, "Excepts unit dimensions for either LHS or RHS shape.");

    if ((dimsLhs.size() - 1) != 1 && (dimsRhs.size() - 1) != 1)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Irregular LHS or RHS shape.");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accmulator type");

    if ((lhsTy.getElementType().isBF16() && !accTy.getElementType().isF32()) ||
        (lhsTy.getElementType().isSignlessInteger(8) &&
         !accTy.getElementType().isSignlessInteger(32)))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only F32 for BF16 or Int32 for Int8 "
                                         "accumulation type is supported.");

    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> dimsAcc;
    llvm::copy_if(accShape, std::back_inserter(dimsAcc),
                  [](int64_t dim) { return dim != 1; });
    if (dimsAcc.size() != 1)
      return rewriter.notifyMatchFailure(contractOp, "Irregular ACC shape");

    auto loc = contractOp.getLoc();
    auto castAcc = vector::ShapeCastOp::create(
        rewriter, loc, VectorType::get(dimsAcc.front(), accTy.getElementType()),
        contractOp.getAcc());

    Value dp;

    if ((dimsRhs.size() - 1) > 0) {
      auto castRhs = vector::ShapeCastOp::create(
          rewriter, loc,
          VectorType::get(dimsRhs.front() * dimsRhs.back(),
                          rhsTy.getElementType()),
          contractOp.getRhs());
      auto castLhs = vector::ShapeCastOp::create(
          rewriter, loc,
          VectorType::get(dimsLhs.front(), lhsTy.getElementType()),
          contractOp.getLhs());
      auto bitcastLhs = vector::BitCastOp::create(
          rewriter, loc, VectorType::get({1}, rewriter.getIntegerType(32)),
          castLhs);
      auto broadcastLhs = vector::BroadcastOp::create(
          rewriter, loc,
          VectorType::get({dimsRhs.front()}, rewriter.getIntegerType(32)),
          bitcastLhs);
      auto bitcastLhsPkType = vector::BitCastOp::create(
          rewriter, loc, castRhs.getResult().getType(), broadcastLhs);

      if (lhsTy.getElementType().isBF16()) {
        dp = x86vector::DotBF16Op::create(
            rewriter, loc,
            VectorType::get(dimsRhs.front(), rewriter.getF32Type()), castAcc,
            bitcastLhsPkType, castRhs);
      }

      if (lhsTy.getElementType().isSignlessInteger(8)) {
        dp = x86vector::DotInt8Op::create(
            rewriter, loc,
            VectorType::get(dimsRhs.front(), rewriter.getIntegerType(32)),
            castAcc, bitcastLhsPkType, castRhs);
      }
    } else {
      auto castLhs = vector::ShapeCastOp::create(
          rewriter, loc,
          VectorType::get(dimsLhs.front() * dimsLhs.back(),
                          lhsTy.getElementType()),
          contractOp.getLhs());
      auto castRhs = vector::ShapeCastOp::create(
          rewriter, loc,
          VectorType::get(dimsRhs.front(), rhsTy.getElementType()),
          contractOp.getRhs());
      auto bitcastRhs = vector::BitCastOp::create(
          rewriter, loc, VectorType::get({1}, rewriter.getIntegerType(32)),
          castRhs);
      auto broadcastRhs = vector::BroadcastOp::create(
          rewriter, loc,
          VectorType::get({dimsLhs.front()}, rewriter.getIntegerType(32)),
          bitcastRhs);
      auto bitcastRhsPkType = vector::BitCastOp::create(
          rewriter, loc, castLhs.getResult().getType(), broadcastRhs);

      if (lhsTy.getElementType().isBF16()) {
        dp = x86vector::DotBF16Op::create(
            rewriter, loc,
            VectorType::get(dimsLhs.front(), rewriter.getF32Type()), castAcc,
            castLhs, bitcastRhsPkType);
      }

      if (lhsTy.getElementType().isSignlessInteger(8)) {
        dp = x86vector::DotInt8Op::create(
            rewriter, loc,
            VectorType::get(dimsLhs.front(), rewriter.getIntegerType(32)),
            castAcc, castLhs, bitcastRhsPkType);
      }
    }

    if (dp) {
      auto castDp = vector::ShapeCastOp::create(rewriter, loc, accTy, dp);
      rewriter.replaceOp(contractOp, castDp);
      return success();
    }

    return failure();
  }
};

void x86vector::populateVectorContractToPackedTypeDotProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToPackedTypeDotProduct>(patterns.getContext());
}
