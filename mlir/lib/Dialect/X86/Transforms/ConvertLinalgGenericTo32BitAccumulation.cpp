//===- ConvertLinalgGenericTo32BitAccumulation.cpp------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

// Rewrites a linalg.generic from low-precision (bf16/i8) to high-precision
// accumulation (f32/i32).
// Performs compute (mul + add) in higher precision, starting from a
// zero-initialized accumulator. Then adds the original output and casts
// (truncates) back to the original type.
//
// Example:
// Input:
// linalg.generic ins(tensor<16x32xbf>, tensor<32x48xbf16>)
// outs(tensor<16x48xbf16) { 	arith.multf : bf16 	arith.addf : bf16 } - >
// tensor<16x48xbf16>
//
// Output:
// linalg.fill ins(f32) outs(tensor<16x48xf32>) -> tensor<16x48xf32>
// linalg.generic ins(tensor<16x32xbf>, tensor<32x48xbf16>)
// outs(tensor<16x48xf32) { 	%a = arith.extf %in : bf16 to f32 	%b = arith.extf
// %in_2 : bf16 to f32 	%c = arith.mulf %a, %b : f32 	arith.addf %out, %c : f32 }
// -> tensor<16x48xf32>
//
// linalg.generic ins(tensor<16x48xf32>, tensor<16x48xbf16>)
// outs(tensor<16x48xbf16>) { 	%a = arith.extf %in_2 : bf16 to f32 	%b =
// arith.addf %in, %a : f32 	%c = arith.truncf %b : f32 to bf16 } ->
// tensor<16x48xbf16>
//
struct ConvertLinalgGenericTo32BitAccumulation
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    if (!genericOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(genericOp,
                                         "Support only for tensor type.");

    if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1)
      return rewriter.notifyMatchFailure(genericOp,
                                         "Needed two input tensors.");

    auto outType =
        llvm::dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());

    if (!outType)
      return rewriter.notifyMatchFailure(genericOp, "No output type detected.");

    if (!outType.getElementType().isBF16() &&
        !outType.getElementType().isSignlessInteger(8))
      return rewriter.notifyMatchFailure(
          genericOp, "The outs type should be BF16 or Int8.");

    Type ipType = rewriter.getBF16Type();
    Type opType = rewriter.getF32Type();

    if (outType.getElementType().isSignlessInteger(8)) {
      ipType = rewriter.getIntegerType(8);
      opType = rewriter.getIntegerType(32);
    }

    if (outType.getElementType().isBF16()) {
      for (Operation &innerOp : genericOp.getRegion().front()) {
        if (isa<arith::MulFOp, arith::AddFOp, linalg::YieldOp>(innerOp))
          continue;

        return rewriter.notifyMatchFailure(
            genericOp,
            "Upsupported operations inside linalg.generic's region.");
      }
    }

    if (outType.getElementType().isSignlessInteger(8)) {
      for (Operation &innerOp : genericOp.getRegion().front()) {
        if (isa<arith::MulIOp, arith::AddIOp, linalg::YieldOp>(innerOp))
          continue;

        return rewriter.notifyMatchFailure(
            genericOp,
            "Upsupported operations inside linalg.generic's region.");
      }
    }

    auto loc = genericOp.getLoc();
    auto tensorType = RankedTensorType::get(outType.getShape(), opType);

    // tensor.empty
    auto empty =
        tensor::EmptyOp::create(rewriter, loc, outType.getShape(), opType);

    auto zeroAttr = rewriter.getFloatAttr(rewriter.getF32Type(), 0.0);
    auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
    if (outType.getElementType().isSignlessInteger(8)) {
      auto zeroAttrI32 =
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0);
      zero = arith::ConstantOp::create(rewriter, loc, zeroAttrI32);
    }

    // fill
    auto fill = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                       ValueRange{empty})
                    .getResult(0);

    // ---- 3. Build new linalg.generic (32 accumulation) ----
    auto newGeneric = linalg::GenericOp::create(
        rewriter, loc,
        tensorType,               // result type
        genericOp.getDpsInputs(), // same inputs
        fill,                     // new init
        genericOp.getIndexingMapsArray(), genericOp.getIteratorTypesArray(),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // args: bf16/I8, bf16/I8, f32/I32
          auto a = args[0];
          auto bval = args[1];
          auto acc = args[2];

          Value sum;
          if (outType.getElementType().isBF16()) {
            // cast inputs
            auto a32 = arith::ExtFOp::create(b, loc, opType, a);
            auto b32 = arith::ExtFOp::create(b, loc, opType, bval);

            // mul + add
            auto mul = arith::MulFOp::create(b, loc, a32, b32);
            sum = arith::AddFOp::create(b, loc, acc, mul);
          }

          if (outType.getElementType().isSignlessInteger(8)) {
            // cast inputs
            auto a32 = arith::ExtSIOp::create(b, loc, opType, a);
            auto b32 = arith::ExtSIOp::create(b, loc, opType, bval);

            // mul + add
            auto mul = arith::MulIOp::create(b, loc, a32, b32);
            sum = arith::AddIOp::create(b, loc, acc, mul);
          }

          linalg::YieldOp::create(b, loc, sum);
        });

    auto outDimSize = outType.getShape().size();

    llvm::SmallVector<utils::IteratorType> iters(outDimSize,
                                                 utils::IteratorType::parallel);

    llvm::ArrayRef<utils::IteratorType> iterRef = iters;

    // ---- 4. Add original output + truncate ----
    auto oldOut = genericOp.getDpsInitOperand(0)->get();
    auto resultType = outType;

    auto finalGeneric = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{newGeneric.getResult(0), oldOut},
        ValueRange{tensor::EmptyOp::create(rewriter, loc, outType.getShape(),
                                           outType.getElementType())},
        llvm::ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(outDimSize),
                                  rewriter.getMultiDimIdentityMap(outDimSize),
                                  rewriter.getMultiDimIdentityMap(outDimSize)},
        iterRef, [&](OpBuilder &b, Location loc, ValueRange args) {
          auto acc = args[0];
          auto accActual = args[1];

          Value cast;
          if (outType.getElementType().isBF16()) {
            auto accActualF32 =
                arith::ExtFOp::create(b, loc, opType, accActual);

            auto sum = arith::AddFOp::create(b, loc, acc, accActualF32);

            cast = arith::TruncFOp::create(b, loc, ipType, sum);
          }

          if (outType.getElementType().isSignlessInteger(8)) {
            auto accActualI32 =
                arith::ExtSIOp::create(b, loc, opType, accActual);

            auto sum = arith::AddIOp::create(b, loc, acc, accActualI32);

            cast = arith::TruncIOp::create(b, loc, ipType, sum);
          }

          linalg::YieldOp::create(b, loc, cast);
        });

    // ---- 5. Replace ----
    rewriter.replaceOp(genericOp, finalGeneric.getResult(0));

    return success();
  }
};

} // namespace

void x86::populateConvertLinalgGenericTo32BitAccumulationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConvertLinalgGenericTo32BitAccumulation>(patterns.getContext());
}
