//===- IntNarrowing.cpp - Integer bitwidth reduction optimizations --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
      if (dyn_cast<IntegerType>(shapedTy.getElementType()))
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

/// Wrapper around `arith::ExtSIOp` and `arith::ExtUIOp` ops that abstracts away
/// the exact op type. Exposes helper functions to query the types, operands,
/// and the result. This is so that we can handle both extension kinds without
/// needing to use templates or branching.
class ExtensionOp {
public:
  /// Attemps to create a new extension op from `op`. Returns an extension op
  /// wrapper when `op` is either `arith.extsi` or `arith.extui`, and failure
  /// otherwise.
  static FailureOr<ExtensionOp> from(Operation *op) {
    if (dyn_cast_or_null<arith::ExtSIOp>(op))
      return ExtensionOp{op, ExtensionKind::Sign};
    if (dyn_cast_or_null<arith::ExtUIOp>(op))
      return ExtensionOp{op, ExtensionKind::Zero};

    return failure();
  }

  ExtensionOp(const ExtensionOp &) = default;
  ExtensionOp &operator=(const ExtensionOp &) = default;

  /// Creates a new extension op of the same kind.
  Operation *recreate(PatternRewriter &rewriter, Location loc, Type newType,
                      Value in) {
    if (kind == ExtensionKind::Sign)
      return rewriter.create<arith::ExtSIOp>(loc, newType, in);

    return rewriter.create<arith::ExtUIOp>(loc, newType, in);
  }

  /// Replaces `toReplace` with a new extension op of the same kind.
  void recreateAndReplace(PatternRewriter &rewriter, Operation *toReplace,
                          Value in) {
    assert(toReplace->getNumResults() == 1);
    Type newType = toReplace->getResult(0).getType();
    Operation *newOp = recreate(rewriter, toReplace->getLoc(), newType, in);
    rewriter.replaceOp(toReplace, newOp->getResult(0));
  }

  ExtensionKind getKind() { return kind; }

  Value getResult() { return op->getResult(0); }
  Value getIn() { return op->getOperand(0); }

  Type getType() { return getResult().getType(); }
  Type getElementType() { return getElementTypeOrSelf(getType()); }
  Type getInType() { return getIn().getType(); }
  Type getInElementType() { return getElementTypeOrSelf(getInType()); }

private:
  ExtensionOp(Operation *op, ExtensionKind kind) : op(op), kind(kind) {
    assert(op);
    assert((isa<arith::ExtSIOp, arith::ExtUIOp>(op)) && "Not an extension op");
  }
  Operation *op = nullptr;
  ExtensionKind kind = {};
};

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

/// Base pattern for arith binary ops.
/// Example:
/// ```
///   %lhs = arith.extsi %a : i8 to i32
///   %rhs = arith.extsi %b : i8 to i32
///   %r = arith.addi %lhs, %rhs : i32
/// ==>
///   %lhs = arith.extsi %a : i8 to i16
///   %rhs = arith.extsi %b : i8 to i16
///   %add = arith.addi %lhs, %rhs : i16
///   %r = arith.extsi %add : i16 to i32
/// ```
template <typename BinaryOp>
struct BinaryOpNarrowingPattern : NarrowingPattern<BinaryOp> {
  using NarrowingPattern<BinaryOp>::NarrowingPattern;

  /// Returns the number of bits required to represent the full result, assuming
  /// that both operands are `operandBits`-wide. Derived classes must implement
  /// this, taking into account `BinaryOp` semantics.
  virtual unsigned getResultBitsProduced(unsigned operandBits) const = 0;

  /// Customization point for patterns that should only apply with
  /// zero/sign-extension ops as arguments.
  virtual bool isSupported(ExtensionOp) const { return true; }

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const final {
    Type origTy = op.getType();
    FailureOr<unsigned> resultBits = calculateBitsRequired(origTy);
    if (failed(resultBits))
      return failure();

    // For the optimization to apply, we expect the lhs to be an extension op,
    // and for the rhs to either be the same extension op or a constant.
    FailureOr<ExtensionOp> ext = ExtensionOp::from(op.getLhs().getDefiningOp());
    if (failed(ext) || !isSupported(*ext))
      return failure();

    FailureOr<unsigned> lhsBitsRequired =
        calculateBitsRequired(ext->getIn(), ext->getKind());
    if (failed(lhsBitsRequired) || *lhsBitsRequired >= *resultBits)
      return failure();

    FailureOr<unsigned> rhsBitsRequired =
        calculateBitsRequired(op.getRhs(), ext->getKind());
    if (failed(rhsBitsRequired) || *rhsBitsRequired >= *resultBits)
      return failure();

    // Negotiate a common bit requirements for both lhs and rhs, accounting for
    // the result requiring more bits than the operands.
    unsigned commonBitsRequired =
        getResultBitsProduced(std::max(*lhsBitsRequired, *rhsBitsRequired));
    FailureOr<Type> narrowTy = this->getNarrowType(commonBitsRequired, origTy);
    if (failed(narrowTy) || calculateBitsRequired(*narrowTy) >= *resultBits)
      return failure();

    Location loc = op.getLoc();
    Value newLhs =
        rewriter.createOrFold<arith::TruncIOp>(loc, *narrowTy, op.getLhs());
    Value newRhs =
        rewriter.createOrFold<arith::TruncIOp>(loc, *narrowTy, op.getRhs());
    Value newAdd = rewriter.create<BinaryOp>(loc, newLhs, newRhs);
    ext->recreateAndReplace(rewriter, op, newAdd);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddIOp Pattern
//===----------------------------------------------------------------------===//

struct AddIPattern final : BinaryOpNarrowingPattern<arith::AddIOp> {
  using BinaryOpNarrowingPattern::BinaryOpNarrowingPattern;

  // Addition may require one extra bit for the result.
  // Example: `UINT8_MAX + 1 == 255 + 1 == 256`.
  unsigned getResultBitsProduced(unsigned operandBits) const override {
    return operandBits + 1;
  }
};

//===----------------------------------------------------------------------===//
// SubIOp Pattern
//===----------------------------------------------------------------------===//

struct SubIPattern final : BinaryOpNarrowingPattern<arith::SubIOp> {
  using BinaryOpNarrowingPattern::BinaryOpNarrowingPattern;

  // This optimization only applies to signed arguments.
  bool isSupported(ExtensionOp ext) const override {
    return ext.getKind() == ExtensionKind::Sign;
  }

  // Subtraction may require one extra bit for the result.
  // Example: `INT8_MAX - (-1) == 127 - (-1) == 128`.
  unsigned getResultBitsProduced(unsigned operandBits) const override {
    return operandBits + 1;
  }
};

//===----------------------------------------------------------------------===//
// MulIOp Pattern
//===----------------------------------------------------------------------===//

struct MulIPattern final : BinaryOpNarrowingPattern<arith::MulIOp> {
  using BinaryOpNarrowingPattern::BinaryOpNarrowingPattern;

  // Multiplication may require up double the operand bits.
  // Example: `UNT8_MAX * UINT8_MAX == 255 * 255 == 65025`.
  unsigned getResultBitsProduced(unsigned operandBits) const override {
    return 2 * operandBits;
  }
};

//===----------------------------------------------------------------------===//
// DivSIOp Pattern
//===----------------------------------------------------------------------===//

struct DivSIPattern final : BinaryOpNarrowingPattern<arith::DivSIOp> {
  using BinaryOpNarrowingPattern::BinaryOpNarrowingPattern;

  // This optimization only applies to signed arguments.
  bool isSupported(ExtensionOp ext) const override {
    return ext.getKind() == ExtensionKind::Sign;
  }

  // Unlike multiplication, signed division requires only one more result bit.
  // Example: `INT8_MIN / (-1) == -128 / (-1) == 128`.
  unsigned getResultBitsProduced(unsigned operandBits) const override {
    return operandBits + 1;
  }
};

//===----------------------------------------------------------------------===//
// DivUIOp Pattern
//===----------------------------------------------------------------------===//

struct DivUIPattern final : BinaryOpNarrowingPattern<arith::DivUIOp> {
  using BinaryOpNarrowingPattern::BinaryOpNarrowingPattern;

  // This optimization only applies to unsigned arguments.
  bool isSupported(ExtensionOp ext) const override {
    return ext.getKind() == ExtensionKind::Zero;
  }

  // Unsigned division does not require any extra result bits.
  unsigned getResultBitsProduced(unsigned operandBits) const override {
    return operandBits;
  }
};

//===----------------------------------------------------------------------===//
// Min/Max Patterns
//===----------------------------------------------------------------------===//

template <typename MinMaxOp, ExtensionKind Kind>
struct MinMaxPattern final : BinaryOpNarrowingPattern<MinMaxOp> {
  using BinaryOpNarrowingPattern<MinMaxOp>::BinaryOpNarrowingPattern;

  bool isSupported(ExtensionOp ext) const override {
    return ext.getKind() == Kind;
  }

  // Min/max returns one of the arguments and does not require any extra result
  // bits.
  unsigned getResultBitsProduced(unsigned operandBits) const override {
    return operandBits;
  }
};
using MaxSIPattern = MinMaxPattern<arith::MaxSIOp, ExtensionKind::Sign>;
using MaxUIPattern = MinMaxPattern<arith::MaxUIOp, ExtensionKind::Zero>;
using MinSIPattern = MinMaxPattern<arith::MinSIOp, ExtensionKind::Sign>;
using MinUIPattern = MinMaxPattern<arith::MinUIOp, ExtensionKind::Zero>;

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
// Index Cast Patterns
//===----------------------------------------------------------------------===//

// These rely on the `ValueBounds` interface for index values. For example, we
// can often statically tell index value bounds of loop induction variables.

template <typename CastOp, ExtensionKind Kind>
struct IndexCastPattern final : NarrowingPattern<CastOp> {
  using NarrowingPattern<CastOp>::NarrowingPattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.getIn();
    // We only support scalar index -> integer casts.
    if (!isa<IndexType>(in.getType()))
      return failure();

    // Check the lower bound in both the signed and unsigned cast case. We
    // conservatively assume that even unsigned casts may be performed on
    // negative indices.
    FailureOr<int64_t> lb = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::LB, in);
    if (failed(lb))
      return failure();

    FailureOr<int64_t> ub = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, in, /*dim=*/std::nullopt,
        /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (failed(ub))
      return failure();

    assert(*lb <= *ub && "Invalid bounds");
    unsigned lbBitsRequired = calculateBitsRequired(APInt(64, *lb), Kind);
    unsigned ubBitsRequired = calculateBitsRequired(APInt(64, *ub), Kind);
    unsigned bitsRequired = std::max(lbBitsRequired, ubBitsRequired);

    IntegerType resultTy = cast<IntegerType>(op.getType());
    if (resultTy.getWidth() <= bitsRequired)
      return failure();

    FailureOr<Type> narrowTy = this->getNarrowType(bitsRequired, resultTy);
    if (failed(narrowTy))
      return failure();

    Value newCast = rewriter.create<CastOp>(op.getLoc(), *narrowTy, op.getIn());

    if (Kind == ExtensionKind::Sign)
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, resultTy, newCast);
    else
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, resultTy, newCast);
    return success();
  }
};
using IndexCastSIPattern =
    IndexCastPattern<arith::IndexCastOp, ExtensionKind::Sign>;
using IndexCastUIPattern =
    IndexCastPattern<arith::IndexCastUIOp, ExtensionKind::Zero>;

//===----------------------------------------------------------------------===//
// Patterns to Commute Extension Ops
//===----------------------------------------------------------------------===//

struct ExtensionOverBroadcast final : NarrowingPattern<vector::BroadcastOp> {
  using NarrowingPattern::NarrowingPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ExtensionOp> ext =
        ExtensionOp::from(op.getSource().getDefiningOp());
    if (failed(ext))
      return failure();

    VectorType origTy = op.getResultVectorType();
    VectorType newTy =
        origTy.cloneWith(origTy.getShape(), ext->getInElementType());
    Value newBroadcast =
        rewriter.create<vector::BroadcastOp>(op.getLoc(), newTy, ext->getIn());
    ext->recreateAndReplace(rewriter, op, newBroadcast);
    return success();
  }
};

struct ExtensionOverExtract final : NarrowingPattern<vector::ExtractOp> {
  using NarrowingPattern::NarrowingPattern;

  LogicalResult matchAndRewrite(vector::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ExtensionOp> ext =
        ExtensionOp::from(op.getVector().getDefiningOp());
    if (failed(ext))
      return failure();

    Value newExtract = rewriter.create<vector::ExtractOp>(
        op.getLoc(), ext->getIn(), op.getMixedPosition());
    ext->recreateAndReplace(rewriter, op, newExtract);
    return success();
  }
};

struct ExtensionOverExtractElement final
    : NarrowingPattern<vector::ExtractElementOp> {
  using NarrowingPattern::NarrowingPattern;

  LogicalResult matchAndRewrite(vector::ExtractElementOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ExtensionOp> ext =
        ExtensionOp::from(op.getVector().getDefiningOp());
    if (failed(ext))
      return failure();

    Value newExtract = rewriter.create<vector::ExtractElementOp>(
        op.getLoc(), ext->getIn(), op.getPosition());
    ext->recreateAndReplace(rewriter, op, newExtract);
    return success();
  }
};

struct ExtensionOverExtractStridedSlice final
    : NarrowingPattern<vector::ExtractStridedSliceOp> {
  using NarrowingPattern::NarrowingPattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ExtensionOp> ext =
        ExtensionOp::from(op.getVector().getDefiningOp());
    if (failed(ext))
      return failure();

    VectorType origTy = op.getType();
    VectorType extractTy =
        origTy.cloneWith(origTy.getShape(), ext->getInElementType());
    Value newExtract = rewriter.create<vector::ExtractStridedSliceOp>(
        op.getLoc(), extractTy, ext->getIn(), op.getOffsets(), op.getSizes(),
        op.getStrides());
    ext->recreateAndReplace(rewriter, op, newExtract);
    return success();
  }
};

/// Base pattern for `vector.insert` narrowing patterns.
template <typename InsertionOp>
struct ExtensionOverInsertionPattern : NarrowingPattern<InsertionOp> {
  using NarrowingPattern<InsertionOp>::NarrowingPattern;

  /// Derived classes must provide a function to create the matching insertion
  /// op based on the original op and new arguments.
  virtual InsertionOp createInsertionOp(PatternRewriter &rewriter,
                                        InsertionOp origInsert,
                                        Value narrowValue,
                                        Value narrowDest) const = 0;

  LogicalResult matchAndRewrite(InsertionOp op,
                                PatternRewriter &rewriter) const final {
    FailureOr<ExtensionOp> ext =
        ExtensionOp::from(op.getSource().getDefiningOp());
    if (failed(ext))
      return failure();

    FailureOr<InsertionOp> newInsert = createNarrowInsert(op, rewriter, *ext);
    if (failed(newInsert))
      return failure();
    ext->recreateAndReplace(rewriter, op, *newInsert);
    return success();
  }

  FailureOr<InsertionOp> createNarrowInsert(InsertionOp op,
                                            PatternRewriter &rewriter,
                                            ExtensionOp insValue) const {
    // Calculate the operand and result bitwidths. We can only apply narrowing
    // when the inserted source value and destination vector require fewer bits
    // than the result. Because the source and destination may have different
    // bitwidths requirements, we have to find the common narrow bitwidth that
    // is greater equal to the operand bitwidth requirements and still narrower
    // than the result.
    FailureOr<unsigned> origBitsRequired = calculateBitsRequired(op.getType());
    if (failed(origBitsRequired))
      return failure();

    // TODO: We could relax this check by disregarding bitwidth requirements of
    // elements that we know will be replaced by the insertion.
    FailureOr<unsigned> destBitsRequired =
        calculateBitsRequired(op.getDest(), insValue.getKind());
    if (failed(destBitsRequired) || *destBitsRequired >= *origBitsRequired)
      return failure();

    FailureOr<unsigned> insertedBitsRequired =
        calculateBitsRequired(insValue.getIn(), insValue.getKind());
    if (failed(insertedBitsRequired) ||
        *insertedBitsRequired >= *origBitsRequired)
      return failure();

    // Find a narrower element type that satisfies the bitwidth requirements of
    // both the source and the destination values.
    unsigned newInsertionBits =
        std::max(*destBitsRequired, *insertedBitsRequired);
    FailureOr<Type> newVecTy =
        this->getNarrowType(newInsertionBits, op.getType());
    if (failed(newVecTy) || *newVecTy == op.getType())
      return failure();

    FailureOr<Type> newInsertedValueTy =
        this->getNarrowType(newInsertionBits, insValue.getType());
    if (failed(newInsertedValueTy))
      return failure();

    Location loc = op.getLoc();
    Value narrowValue = rewriter.createOrFold<arith::TruncIOp>(
        loc, *newInsertedValueTy, insValue.getResult());
    Value narrowDest =
        rewriter.createOrFold<arith::TruncIOp>(loc, *newVecTy, op.getDest());
    return createInsertionOp(rewriter, op, narrowValue, narrowDest);
  }
};

struct ExtensionOverInsert final
    : ExtensionOverInsertionPattern<vector::InsertOp> {
  using ExtensionOverInsertionPattern::ExtensionOverInsertionPattern;

  vector::InsertOp createInsertionOp(PatternRewriter &rewriter,
                                     vector::InsertOp origInsert,
                                     Value narrowValue,
                                     Value narrowDest) const override {
    return rewriter.create<vector::InsertOp>(origInsert.getLoc(), narrowValue,
                                             narrowDest,
                                             origInsert.getMixedPosition());
  }
};

struct ExtensionOverInsertElement final
    : ExtensionOverInsertionPattern<vector::InsertElementOp> {
  using ExtensionOverInsertionPattern::ExtensionOverInsertionPattern;

  vector::InsertElementOp createInsertionOp(PatternRewriter &rewriter,
                                            vector::InsertElementOp origInsert,
                                            Value narrowValue,
                                            Value narrowDest) const override {
    return rewriter.create<vector::InsertElementOp>(
        origInsert.getLoc(), narrowValue, narrowDest, origInsert.getPosition());
  }
};

struct ExtensionOverInsertStridedSlice final
    : ExtensionOverInsertionPattern<vector::InsertStridedSliceOp> {
  using ExtensionOverInsertionPattern::ExtensionOverInsertionPattern;

  vector::InsertStridedSliceOp
  createInsertionOp(PatternRewriter &rewriter,
                    vector::InsertStridedSliceOp origInsert, Value narrowValue,
                    Value narrowDest) const override {
    return rewriter.create<vector::InsertStridedSliceOp>(
        origInsert.getLoc(), narrowValue, narrowDest, origInsert.getOffsets(),
        origInsert.getStrides());
  }
};

struct ExtensionOverShapeCast final : NarrowingPattern<vector::ShapeCastOp> {
  using NarrowingPattern::NarrowingPattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ExtensionOp> ext =
        ExtensionOp::from(op.getSource().getDefiningOp());
    if (failed(ext))
      return failure();

    VectorType origTy = op.getResultVectorType();
    VectorType newTy =
        origTy.cloneWith(origTy.getShape(), ext->getInElementType());
    Value newCast =
        rewriter.create<vector::ShapeCastOp>(op.getLoc(), newTy, ext->getIn());
    ext->recreateAndReplace(rewriter, op, newCast);
    return success();
  }
};

struct ExtensionOverTranspose final : NarrowingPattern<vector::TransposeOp> {
  using NarrowingPattern::NarrowingPattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ExtensionOp> ext =
        ExtensionOp::from(op.getVector().getDefiningOp());
    if (failed(ext))
      return failure();

    VectorType origTy = op.getResultVectorType();
    VectorType newTy =
        origTy.cloneWith(origTy.getShape(), ext->getInElementType());
    Value newTranspose = rewriter.create<vector::TransposeOp>(
        op.getLoc(), newTy, ext->getIn(), op.getPermutation());
    ext->recreateAndReplace(rewriter, op, newTranspose);
    return success();
  }
};

struct ExtensionOverFlatTranspose final
    : NarrowingPattern<vector::FlatTransposeOp> {
  using NarrowingPattern::NarrowingPattern;

  LogicalResult matchAndRewrite(vector::FlatTransposeOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ExtensionOp> ext =
        ExtensionOp::from(op.getMatrix().getDefiningOp());
    if (failed(ext))
      return failure();

    VectorType origTy = op.getType();
    VectorType newTy =
        origTy.cloneWith(origTy.getShape(), ext->getInElementType());
    Value newTranspose = rewriter.create<vector::FlatTransposeOp>(
        op.getLoc(), newTy, ext->getIn(), op.getRowsAttr(),
        op.getColumnsAttr());
    ext->recreateAndReplace(rewriter, op, newTranspose);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definitions
//===----------------------------------------------------------------------===//

struct ArithIntNarrowingPass final
    : impl::ArithIntNarrowingBase<ArithIntNarrowingPass> {
  using ArithIntNarrowingBase::ArithIntNarrowingBase;

  void runOnOperation() override {
    if (bitwidthsSupported.empty() ||
        llvm::is_contained(bitwidthsSupported, 0)) {
      // Invalid pass options.
      return signalPassFailure();
    }

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
  patterns.add<ExtensionOverBroadcast, ExtensionOverExtract,
               ExtensionOverExtractElement, ExtensionOverExtractStridedSlice,
               ExtensionOverInsert, ExtensionOverInsertElement,
               ExtensionOverInsertStridedSlice, ExtensionOverShapeCast,
               ExtensionOverTranspose, ExtensionOverFlatTranspose>(
      patterns.getContext(), options, PatternBenefit(2));

  patterns.add<AddIPattern, SubIPattern, MulIPattern, DivSIPattern,
               DivUIPattern, MaxSIPattern, MaxUIPattern, MinSIPattern,
               MinUIPattern, SIToFPPattern, UIToFPPattern, IndexCastSIPattern,
               IndexCastUIPattern>(patterns.getContext(), options);
}

} // namespace mlir::arith
