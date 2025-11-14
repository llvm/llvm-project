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

    if (contractOp.getKind() != vector::CombiningKind::ADD) {
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind");
    }

    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only F32 lowering is supported.");
    if (llvm::any_of(lhsTy.getShape(), [](int64_t dim) { return dim != 1; }))
      return rewriter.notifyMatchFailure(
          contractOp, "Expects one for all dimensions of LHS");

    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    llvm::SmallVector<int64_t> dimsRhs;
    llvm::copy_if(rhsShape, std::back_inserter(dimsRhs),
                  [](int64_t dim) { return dim != 1; });
    if (dimsRhs.size() != 1)
      return rewriter.notifyMatchFailure(contractOp, "Irregular RHS shape");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    assert(accTy && "Invalid accumulator");
    ArrayRef<int64_t> accShape = accTy.getShape();
    llvm::SmallVector<int64_t> dimsAcc;
    llvm::copy_if(accShape, std::back_inserter(dimsAcc),
                  [](int64_t dim) { return dim != 1; });
    if (dimsAcc.size() != 1)
      return rewriter.notifyMatchFailure(contractOp, "Irregular ACC shape");

    // Lowers vector.contract into a broadcast+FMA sequence.
    auto loc = contractOp.getLoc();
    auto castLhs = vector::ShapeCastOp::create(
        rewriter, loc, VectorType::get(1, lhsTy.getElementType()),
        contractOp.getLhs());
    auto castRhs = vector::ShapeCastOp::create(
        rewriter, loc, VectorType::get(dimsRhs.front(), rhsTy.getElementType()),
        contractOp.getRhs());
    auto castAcc = vector::ShapeCastOp::create(
        rewriter, loc, VectorType::get(dimsAcc.front(), accTy.getElementType()),
        contractOp.getAcc());
    auto broadcastLhs = vector::BroadcastOp::create(
        rewriter, loc, castRhs.getResult().getType(), castLhs);
    auto fma =
        vector::FMAOp::create(rewriter, loc, broadcastLhs, castRhs, castAcc);
    auto castFma = vector::ShapeCastOp::create(rewriter, loc, accTy, fma);

    rewriter.replaceOp(contractOp, castFma);

    return success();
  }
};

void x86vector::populateVectorContractToFMAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToFMA>(patterns.getContext());
}
