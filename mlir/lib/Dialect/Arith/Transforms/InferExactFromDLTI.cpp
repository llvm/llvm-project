//===- InferExactFromDLTI.cpp - Infer exact flags from DLTI ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir {
namespace arith {
#define GEN_PASS_DEF_ARITHINFEREXACTFROMDLTI
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace arith
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;

static unsigned getBitwidth(Type type, unsigned indexBitwidth) {
  Type elemType = getElementTypeOrSelf(type);
  if (isa<IndexType>(elemType))
    return indexBitwidth;
  return elemType.getIntOrFloatBitWidth();
}

namespace {
template <typename CastOp>
struct InferExactOnIndexCast final : OpRewritePattern<CastOp> {
  InferExactOnIndexCast(MLIRContext *context, unsigned indexBitwidth)
      : OpRewritePattern<CastOp>(context), indexBitwidth(indexBitwidth) {}

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getExact())
      return failure();

    unsigned srcBW = getBitwidth(op.getIn().getType(), indexBitwidth);
    unsigned dstBW = getBitwidth(op.getType(), indexBitwidth);
    if (srcBW > dstBW)
      return rewriter.notifyMatchFailure(op, "source is wider than dest");

    rewriter.modifyOpInPlace(op, [&] { op.setExact(true); });
    return success();
  }

private:
  unsigned indexBitwidth;
};

struct ArithInferExactFromDLTIPass
    : public arith::impl::ArithInferExactFromDLTIBase<
          ArithInferExactFromDLTIPass> {

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    DataLayout layout = DataLayout::closest(op);
    unsigned indexBitwidth = layout.getTypeSizeInBits(IndexType::get(ctx));

    RewritePatternSet patterns(ctx);
    populateInferExactFromDLTIPatterns(patterns, indexBitwidth);

    walkAndApplyPatterns(op, std::move(patterns));
  }
};
} // end anonymous namespace

void mlir::arith::populateInferExactFromDLTIPatterns(
    RewritePatternSet &patterns, unsigned indexBitwidth) {
  patterns.add<InferExactOnIndexCast<IndexCastOp>,
               InferExactOnIndexCast<IndexCastUIOp>>(patterns.getContext(),
                                                     indexBitwidth);
}
