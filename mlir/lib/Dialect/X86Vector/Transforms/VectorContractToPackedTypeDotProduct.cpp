//===- VectorContractToPackedTypeDotProduct.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

    if (contractOp.getKind() != vector::CombiningKind::ADD) {
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind");
    }

    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16() &&
        !lhsTy.getElementType().isSignlessInteger(8))
      return rewriter.notifyMatchFailure(
          contractOp, "Only BF16/Int8 lowering is supported.");
    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    if (lhsTy.getElementType().isBF16() && lhsShape.back() != 2)
      return rewriter.notifyMatchFailure(
          contractOp, "The LHS vnni dim should be 2 for BF16.");

    if (lhsTy.getElementType().isSignlessInteger(8) && lhsShape.back() != 4)
      return rewriter.notifyMatchFailure(
          contractOp, "The LHS vnni dim should be 4 for Int8.");
    llvm::SmallVector<int64_t> dimsLhs;
    llvm::copy_if(lhsShape, std::back_inserter(dimsLhs),
                  [](int64_t dim) { return dim != 1; });
    if (dimsLhs.size() != 1)
      return rewriter.notifyMatchFailure(contractOp, "Irregular LHS shape");

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    if (lhsTy.getElementType().isBF16() && rhsShape.back() != 2)
      return rewriter.notifyMatchFailure(
          contractOp, "The RHS vnni dim should be 2 for BF16.");
    if (lhsTy.getElementType().isSignlessInteger(8) && rhsShape.back() != 4)
      return rewriter.notifyMatchFailure(
          contractOp, "The RHS vnni dim should be 4 for Int8.");
    llvm::SmallVector<int64_t> dimsRhs;
    llvm::copy_if(rhsShape, std::back_inserter(dimsRhs),
                  [](int64_t dim) { return dim != 1; });
    if (dimsRhs.size() != 2)
      return rewriter.notifyMatchFailure(contractOp, "Irregular RHS shape");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    assert(accTy && "Invalid accumulator");
    if (!accTy.getElementType().isF32() &&
        !accTy.getElementType().isSignlessInteger(32))
      return rewriter.notifyMatchFailure(
          contractOp, "Only F32/Int32 accumulation is supported.");
    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> dimsAcc;
    llvm::copy_if(accShape, std::back_inserter(dimsAcc),
                  [](int64_t dim) { return dim != 1; });
    if (dimsAcc.size() != 1)
      return rewriter.notifyMatchFailure(contractOp, "Irregular ACC shape");

    auto loc = contractOp.getLoc();
    auto castRhs = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(dimsRhs.front() * dimsRhs.back(),
                        rhsTy.getElementType()),
        contractOp.getRhs());

    auto castAcc = vector::ShapeCastOp::create(
        rewriter, loc, VectorType::get(dimsAcc.front(), accTy.getElementType()),
        contractOp.getAcc());

    auto castLhs = vector::ShapeCastOp::create(
        rewriter, loc, VectorType::get(dimsLhs.front(), lhsTy.getElementType()),
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

    Value dp;

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
