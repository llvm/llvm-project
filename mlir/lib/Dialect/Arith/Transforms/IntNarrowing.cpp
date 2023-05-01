//===- IntNarrowing.cpp - Integer bitwidth reduction optimizations --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <cstdint>

namespace mlir::arith {
#define GEN_PASS_DEF_ARITHINTNARROWING
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"
} // namespace mlir::arith

namespace mlir::arith {
namespace {
//===----------------------------------------------------------------------===//
// Common Helpers
//===----------------------------------------------------------------------===//

/// The base for integer bitwidth narrowing patterns.
template <typename SourceOp>
struct NarrowingPattern : OpRewritePattern<SourceOp> {
  NarrowingPattern(MLIRContext *ctx, const ArithIntNarrowingOptions &options,
                   PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(ctx, benefit),
        supportedBitwidths(options.bitwidthsSupported.begin(),
                           options.bitwidthsSupported.end()) {
    assert(!supportedBitwidths.empty() && "Invalid options");
    assert(!llvm::is_contained(supportedBitwidths, 0) && "Invalid bitwidth");
    llvm::sort(supportedBitwidths);
  }

  FailureOr<unsigned>
  getNarrowestCompatibleBitwidth(unsigned bitsRequired) const {
    for (unsigned candidate : supportedBitwidths)
      if (candidate >= bitsRequired)
        return candidate;

    return failure();
  }

  /// Returns the narrowest supported type that fits `bitsRequired`.
  FailureOr<Type> getNarrowType(unsigned bitsRequired, Type origTy) const {
    assert(origTy);
    FailureOr<unsigned> bestBitwidth =
        getNarrowestCompatibleBitwidth(bitsRequired);
    if (failed(bestBitwidth))
      return failure();

    Type elemTy = getElementTypeOrSelf(origTy);
    if (!isa<IntegerType>(elemTy))
      return failure();

    auto newElemTy = IntegerType::get(origTy.getContext(), bitsRequired);
    if (newElemTy == elemTy)
      return failure();

    if (origTy == elemTy)
      return newElemTy;

    if (auto shapedTy = dyn_cast<ShapedType>(origTy))
      if (auto elemTy = dyn_cast<IntegerType>(shapedTy.getElementType()))
        return shapedTy.clone(shapedTy.getShape(), newElemTy);

    return failure();
  }

private:
  // Supported integer bitwidths in the ascending order.
  llvm::SmallVector<unsigned, 6> supportedBitwidths;
};

/// Returns the integer bitwidth required to represent `type`.
FailureOr<unsigned> calculateBitsRequired(Type type) {
  assert(type);
  if (auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(type)))
    return intTy.getWidth();

  return failure();
}

enum class ExtensionKind { Sign, Zero };

/// Returns the integer bitwidth required to represent `value`.
/// Looks through either sign- or zero-extension as specified by
/// `lookThroughExtension`.
FailureOr<unsigned> calculateBitsRequired(Value value,
                                          ExtensionKind lookThroughExtension) {
  if (lookThroughExtension == ExtensionKind::Sign) {
    if (auto sext = value.getDefiningOp<arith::ExtSIOp>())
      return calculateBitsRequired(sext.getIn().getType());
  } else if (lookThroughExtension == ExtensionKind::Zero) {
    if (auto zext = value.getDefiningOp<arith::ExtUIOp>())
      return calculateBitsRequired(zext.getIn().getType());
  }

  // If nothing else worked, return the type requirements for this element type.
  return calculateBitsRequired(value.getType());
}

//===----------------------------------------------------------------------===//
// *IToFPOp Patterns
//===----------------------------------------------------------------------===//

template <typename IToFPOp, ExtensionKind Extension>
struct IToFPPattern final : NarrowingPattern<IToFPOp> {
  using NarrowingPattern<IToFPOp>::NarrowingPattern;

  LogicalResult matchAndRewrite(IToFPOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<unsigned> narrowestWidth =
        calculateBitsRequired(op.getIn(), Extension);
    if (failed(narrowestWidth))
      return failure();

    FailureOr<Type> narrowTy =
        this->getNarrowType(*narrowestWidth, op.getIn().getType());
    if (failed(narrowTy))
      return failure();

    Value newIn = rewriter.createOrFold<arith::TruncIOp>(op.getLoc(), *narrowTy,
                                                         op.getIn());
    rewriter.replaceOpWithNewOp<IToFPOp>(op, op.getType(), newIn);
    return success();
  }
};
using SIToFPPattern = IToFPPattern<arith::SIToFPOp, ExtensionKind::Sign>;
using UIToFPPattern = IToFPPattern<arith::UIToFPOp, ExtensionKind::Zero>;

//===----------------------------------------------------------------------===//
// Patterns to Commute Extension Ops
//===----------------------------------------------------------------------===//

struct ExtensionOverExtract final : OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    Operation *def = op.getVector().getDefiningOp();
    if (!def)
      return failure();

    return TypeSwitch<Operation *, LogicalResult>(def)
        .Case<arith::ExtSIOp, arith::ExtUIOp>([&](auto extOp) {
          Value newExtract = rewriter.create<vector::ExtractOp>(
              op.getLoc(), extOp.getIn(), op.getPosition());
          rewriter.replaceOpWithNewOp<decltype(extOp)>(op, op.getType(),
                                                       newExtract);
          return success();
        })
        .Default(failure());
  }
};

struct ExtensionOverExtractElement final
    : OpRewritePattern<vector::ExtractElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractElementOp op,
                                PatternRewriter &rewriter) const override {
    Operation *def = op.getVector().getDefiningOp();
    if (!def)
      return failure();

    return TypeSwitch<Operation *, LogicalResult>(def)
        .Case<arith::ExtSIOp, arith::ExtUIOp>([&](auto extOp) {
          Value newExtract = rewriter.create<vector::ExtractElementOp>(
              op.getLoc(), extOp.getIn(), op.getPosition());
          rewriter.replaceOpWithNewOp<decltype(extOp)>(op, op.getType(),
                                                       newExtract);
          return success();
        })
        .Default(failure());
  }
};

struct ExtensionOverExtractStridedSlice final
    : OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    Operation *def = op.getVector().getDefiningOp();
    if (!def)
      return failure();

    return TypeSwitch<Operation *, LogicalResult>(def)
        .Case<arith::ExtSIOp, arith::ExtUIOp>([&](auto extOp) {
          VectorType origTy = op.getType();
          Type inElemTy =
              cast<VectorType>(extOp.getIn().getType()).getElementType();
          VectorType extractTy = origTy.cloneWith(origTy.getShape(), inElemTy);
          Value newExtract = rewriter.create<vector::ExtractStridedSliceOp>(
              op.getLoc(), extractTy, extOp.getIn(), op.getOffsets(),
              op.getSizes(), op.getStrides());
          rewriter.replaceOpWithNewOp<decltype(extOp)>(op, op.getType(),
                                                       newExtract);
          return success();
        })
        .Default(failure());
  }
};

//===----------------------------------------------------------------------===//
// Pass Definitions
//===----------------------------------------------------------------------===//

struct ArithIntNarrowingPass final
    : impl::ArithIntNarrowingBase<ArithIntNarrowingPass> {
  using ArithIntNarrowingBase::ArithIntNarrowingBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();
    RewritePatternSet patterns(ctx);
    populateArithIntNarrowingPatterns(
        patterns, ArithIntNarrowingOptions{bitwidthsSupported});
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void populateArithIntNarrowingPatterns(
    RewritePatternSet &patterns, const ArithIntNarrowingOptions &options) {
  // Add commute patterns with a higher benefit. This is to expose more
  // optimization opportunities to narrowing patterns.
  patterns.add<ExtensionOverExtract, ExtensionOverExtractElement,
               ExtensionOverExtractStridedSlice>(patterns.getContext(),
                                                 PatternBenefit(2));

  patterns.add<SIToFPPattern, UIToFPPattern>(patterns.getContext(), options);
}

} // namespace mlir::arith
