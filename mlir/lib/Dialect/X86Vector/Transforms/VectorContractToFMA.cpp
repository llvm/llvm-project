//===- VectorContractToFMA.cpp --------------------------------------------===//
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

namespace {

// Implements outer product contraction as a sequence of broadcast and
// FMA operations.
//
// For example - for F32 type:
// ```
//   vector.contract <1x1xf32>, <1x16xf32> into <1x16xf32>
// ```
// to
// ```
//   vector.broadcast %lhs to <16xf32>
//   vector.fma vector<16xf32>
// ```
struct VectorContractToFMA : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only F32 lowering is supported.");

    ArrayRef<int64_t> lhsShape = lhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimLhs;
    llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                  [](int64_t dim) { return dim != 1; });

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimRhs;
    llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                  [](int64_t dim) { return dim != 1; });

    if (nonUnitDimLhs.size() > 0 && nonUnitDimRhs.size() > 0)
      return rewriter.notifyMatchFailure(
          contractOp, "Excepts unit dimensions for either LHS or RHS shape.");

    if (nonUnitDimLhs.size() != 1 && nonUnitDimRhs.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp,
          "Excepts a one non-unit A/B dimension for either LHS or RHS shape.");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Accmulator is not a vector type");

    if (!accTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Accmulator should be F32 type.");

    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> nonUnitDimAcc;
    llvm::copy_if(accShape, std::back_inserter(nonUnitDimAcc),
                  [](int64_t dim) { return dim != 1; });
    if (nonUnitDimAcc.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "A or B dimension should be non-unit.");

    // Lowers vector.contract into a broadcast+FMA sequence.
    auto loc = contractOp.getLoc();
    auto castAcc = vector::ShapeCastOp::create(
        rewriter, loc,
        VectorType::get(nonUnitDimAcc.front(), accTy.getElementType()),
        contractOp.getAcc());

    vector::FMAOp fma;

    // Broadcast the unit-dimension LHS or RHS to match the vector length of the
    // corresponding non-unit dimension on the other operand. For example,
    // if LHS has type vector<1x1xf32> and RHS has type vector<1x16xf32>, we
    // broadcast the LHS to vector<1x16xf32>. In the opposite case (non-unit
    // dimension on the LHS), we broadcast the RHS instead.
    if (nonUnitDimRhs.size() > 0) {
      auto castLhs = vector::ShapeCastOp::create(
          rewriter, loc, VectorType::get(1, lhsTy.getElementType()),
          contractOp.getLhs());
      auto castRhs = vector::ShapeCastOp::create(
          rewriter, loc,
          VectorType::get(nonUnitDimRhs.front(), rhsTy.getElementType()),
          contractOp.getRhs());
      auto broadcastLhs = vector::BroadcastOp::create(
          rewriter, loc, castRhs.getResult().getType(), castLhs);
      fma =
          vector::FMAOp::create(rewriter, loc, broadcastLhs, castRhs, castAcc);
    } else {
      auto castLhs = vector::ShapeCastOp::create(
          rewriter, loc,
          VectorType::get(nonUnitDimLhs.front(), lhsTy.getElementType()),
          contractOp.getLhs());
      auto castRhs = vector::ShapeCastOp::create(
          rewriter, loc, VectorType::get(1, rhsTy.getElementType()),
          contractOp.getRhs());
      auto broadcastRhs = vector::BroadcastOp::create(
          rewriter, loc, castLhs.getResult().getType(), castRhs);
      fma =
          vector::FMAOp::create(rewriter, loc, castLhs, broadcastRhs, castAcc);
    }

    auto castFma = vector::ShapeCastOp::create(rewriter, loc, accTy, fma);
    rewriter.replaceOp(contractOp, castFma);

    return success();
  }
};

} // namespace

void x86vector::populateVectorContractToFMAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToFMA>(patterns.getContext());
}
