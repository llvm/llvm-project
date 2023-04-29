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

    auto newElemTy = IntegerType::get(origTy.getContext(), *bestBitwidth);
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

ExtensionKind getExtensionKind(Operation *op) {
  assert(op);
  assert((isa<arith::ExtSIOp, arith::ExtUIOp>(op)) && "Not an extension op");
  return isa<arith::ExtSIOp>(op) ? ExtensionKind::Sign : ExtensionKind::Zero;
}

/// Returns the integer bitwidth required to represent `value`.
unsigned calculateBitsRequired(const APInt &value,
                               ExtensionKind lookThroughExtension) {
  // For unsigned values, we only need the active bits. As a special case, zero
  // requires one bit.
  if (lookThroughExtension == ExtensionKind::Zero)
    return std::max(value.getActiveBits(), 1u);

  // If a signed value is nonnegative, we need one extra bit for the sign.
  if (value.isNonNegative())
    return value.getActiveBits() + 1;

  // For the signed min, we need all the bits.
  if (value.isMinSignedValue())
    return value.getBitWidth();

  // For negative values, we need all the non-sign bits and one extra bit for
  // the sign.
  return value.getBitWidth() - value.getNumSignBits() + 1;
}

/// Returns the integer bitwidth required to represent `value`.
/// Looks through either sign- or zero-extension as specified by
/// `lookThroughExtension`.
FailureOr<unsigned> calculateBitsRequired(Value value,
                                          ExtensionKind lookThroughExtension) {
  // Handle constants.
  if (TypedAttr attr; matchPattern(value, m_Constant(&attr))) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return calculateBitsRequired(intAttr.getValue(), lookThroughExtension);

    if (auto elemsAttr = dyn_cast<DenseElementsAttr>(attr)) {
      if (elemsAttr.getElementType().isIntOrIndex()) {
        if (elemsAttr.isSplat())
          return calculateBitsRequired(elemsAttr.getSplatValue<APInt>(),
                                       lookThroughExtension);

        unsigned maxBits = 1;
        for (const APInt &elemValue : elemsAttr.getValues<APInt>())
          maxBits = std::max(
              maxBits, calculateBitsRequired(elemValue, lookThroughExtension));
        return maxBits;
      }
    }
  }

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

struct ExtensionOverExtract final : NarrowingPattern<vector::ExtractOp> {
  using NarrowingPattern::NarrowingPattern;

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
    : NarrowingPattern<vector::ExtractElementOp> {
  using NarrowingPattern::NarrowingPattern;

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
    : NarrowingPattern<vector::ExtractStridedSliceOp> {
  using NarrowingPattern::NarrowingPattern;

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

struct ExtensionOverInsert final : NarrowingPattern<vector::InsertOp> {
  using NarrowingPattern::NarrowingPattern;

  LogicalResult matchAndRewrite(vector::InsertOp op,
                                PatternRewriter &rewriter) const override {
    Operation *def = op.getSource().getDefiningOp();
    if (!def)
      return failure();

    return TypeSwitch<Operation *, LogicalResult>(def)
        .Case<arith::ExtSIOp, arith::ExtUIOp>([&](auto extOp) {
          // Rewrite the insertion in terms of narrower operands
          // and later extend the result to the original bitwidth.
          FailureOr<vector::InsertOp> newInsert =
              createNarrowInsert(op, rewriter, extOp);
          if (failed(newInsert))
            return failure();
          rewriter.replaceOpWithNewOp<decltype(extOp)>(op, op.getType(),
                                                       *newInsert);
          return success();
        })
        .Default(failure());
  }

  FailureOr<vector::InsertOp> createNarrowInsert(vector::InsertOp op,
                                                 PatternRewriter &rewriter,
                                                 Operation *insValue) const {
    assert((isa<arith::ExtSIOp, arith::ExtUIOp>(insValue)));

    // Calculate the operand and result bitwidths. We can only apply narrowing
    // when the inserted source value and destination vector require fewer bits
    // than the result. Because the source and destination may have different
    // bitwidths requirements, we have to find the common narrow bitwidth that
    // is greater equal to the operand bitwidth requirements and still narrower
    // than the result.
    FailureOr<unsigned> origBitsRequired = calculateBitsRequired(op.getType());
    if (failed(origBitsRequired))
      return failure();

    ExtensionKind kind = getExtensionKind(insValue);
    FailureOr<unsigned> destBitsRequired =
        calculateBitsRequired(op.getDest(), kind);
    if (failed(destBitsRequired) || *destBitsRequired >= *origBitsRequired)
      return failure();

    FailureOr<unsigned> insertedBitsRequired =
        calculateBitsRequired(insValue->getOperands().front(), kind);
    if (failed(insertedBitsRequired) ||
        *insertedBitsRequired >= *origBitsRequired)
      return failure();

    // Find a narrower element type that satisfies the bitwidth requirements of
    // both the source and the destination values.
    unsigned newInsertionBits =
        std::max(*destBitsRequired, *insertedBitsRequired);
    FailureOr<Type> newVecTy = getNarrowType(newInsertionBits, op.getType());
    if (failed(newVecTy) || *newVecTy == op.getType())
      return failure();

    FailureOr<Type> newInsertedValueTy =
        getNarrowType(newInsertionBits, insValue->getResultTypes().front());
    if (failed(newInsertedValueTy))
      return failure();

    Location loc = op.getLoc();
    Value narrowValue = rewriter.createOrFold<arith::TruncIOp>(
        loc, *newInsertedValueTy, insValue->getResult(0));
    Value narrowDest =
        rewriter.createOrFold<arith::TruncIOp>(loc, *newVecTy, op.getDest());
    return rewriter.create<vector::InsertOp>(loc, narrowValue, narrowDest,
                                             op.getPosition());
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
               ExtensionOverExtractStridedSlice, ExtensionOverInsert>(
      patterns.getContext(), options, PatternBenefit(2));

  patterns.add<SIToFPPattern, UIToFPPattern>(patterns.getContext(), options);
}

} // namespace mlir::arith
