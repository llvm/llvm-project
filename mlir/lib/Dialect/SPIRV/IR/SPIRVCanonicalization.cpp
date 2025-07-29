//===- SPIRVCanonicalization.cpp - MLIR SPIR-V canonicalization patterns --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the folders and canonicalization patterns for SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <utility>

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

/// Returns the boolean value under the hood if the given `boolAttr` is a scalar
/// or splat vector bool constant.
static std::optional<bool> getScalarOrSplatBoolAttr(Attribute attr) {
  if (!attr)
    return std::nullopt;

  if (auto boolAttr = llvm::dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue();
  if (auto splatAttr = llvm::dyn_cast<SplatElementsAttr>(attr))
    if (splatAttr.getElementType().isInteger(1))
      return splatAttr.getSplatValue<bool>();
  return std::nullopt;
}

// Extracts an element from the given `composite` by following the given
// `indices`. Returns a null Attribute if error happens.
static Attribute extractCompositeElement(Attribute composite,
                                         ArrayRef<unsigned> indices) {
  // Check that given composite is a constant.
  if (!composite)
    return {};
  // Return composite itself if we reach the end of the index chain.
  if (indices.empty())
    return composite;

  if (auto vector = llvm::dyn_cast<ElementsAttr>(composite)) {
    assert(indices.size() == 1 && "must have exactly one index for a vector");
    return vector.getValues<Attribute>()[indices[0]];
  }

  if (auto array = llvm::dyn_cast<ArrayAttr>(composite)) {
    assert(!indices.empty() && "must have at least one index for an array");
    return extractCompositeElement(array.getValue()[indices[0]],
                                   indices.drop_front());
  }

  return {};
}

static bool isDivZeroOrOverflow(const APInt &a, const APInt &b) {
  bool div0 = b.isZero();
  bool overflow = a.isMinSignedValue() && b.isAllOnes();

  return div0 || overflow;
}

//===----------------------------------------------------------------------===//
// TableGen'erated canonicalizers
//===----------------------------------------------------------------------===//

namespace {
#include "SPIRVCanonicalization.inc"
} // namespace

//===----------------------------------------------------------------------===//
// spirv.AccessChainOp
//===----------------------------------------------------------------------===//

namespace {

/// Combines chained `spirv::AccessChainOp` operations into one
/// `spirv::AccessChainOp` operation.
struct CombineChainedAccessChain final
    : OpRewritePattern<spirv::AccessChainOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::AccessChainOp accessChainOp,
                                PatternRewriter &rewriter) const override {
    auto parentAccessChainOp =
        accessChainOp.getBasePtr().getDefiningOp<spirv::AccessChainOp>();

    if (!parentAccessChainOp) {
      return failure();
    }

    // Combine indices.
    SmallVector<Value, 4> indices(parentAccessChainOp.getIndices());
    llvm::append_range(indices, accessChainOp.getIndices());

    rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
        accessChainOp, parentAccessChainOp.getBasePtr(), indices);

    return success();
  }
};
} // namespace

void spirv::AccessChainOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<CombineChainedAccessChain>(context);
}

//===----------------------------------------------------------------------===//
// spirv.IAddCarry
//===----------------------------------------------------------------------===//

// We are required to use CompositeConstructOp to create a constant struct as
// they are not yet implemented as constant, hence we can not do so in a fold.
struct IAddCarryFold final : OpRewritePattern<spirv::IAddCarryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::IAddCarryOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();
    Type constituentType = lhs.getType();

    // iaddcarry (x, 0) = <0, x>
    if (matchPattern(rhs, m_Zero())) {
      Value constituents[2] = {rhs, lhs};
      rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, op.getType(),
                                                               constituents);
      return success();
    }

    // According to the SPIR-V spec:
    //
    //  Result Type must be from OpTypeStruct.  The struct must have two
    //  members...
    //
    //  Member 0 of the result gets the low-order bits (full component width) of
    //  the addition.
    //
    //  Member 1 of the result gets the high-order (carry) bit of the result of
    //  the addition. That is, it gets the value 1 if the addition overflowed
    //  the component width, and 0 otherwise.
    Attribute lhsAttr;
    Attribute rhsAttr;
    if (!matchPattern(lhs, m_Constant(&lhsAttr)) ||
        !matchPattern(rhs, m_Constant(&rhsAttr)))
      return failure();

    auto adds = constFoldBinaryOp<IntegerAttr>(
        {lhsAttr, rhsAttr},
        [](const APInt &a, const APInt &b) { return a + b; });
    if (!adds)
      return failure();

    auto carrys = constFoldBinaryOp<IntegerAttr>(
        ArrayRef{adds, lhsAttr}, [](const APInt &a, const APInt &b) {
          APInt zero = APInt::getZero(a.getBitWidth());
          return a.ult(b) ? (zero + 1) : zero;
        });

    if (!carrys)
      return failure();

    Value addsVal =
        spirv::ConstantOp::create(rewriter, loc, constituentType, adds);

    Value carrysVal =
        spirv::ConstantOp::create(rewriter, loc, constituentType, carrys);

    // Create empty struct
    Value undef = spirv::UndefOp::create(rewriter, loc, op.getType());
    // Fill in adds at id 0
    Value intermediate =
        spirv::CompositeInsertOp::create(rewriter, loc, addsVal, undef, 0);
    // Fill in carrys at id 1
    rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(op, carrysVal,
                                                          intermediate, 1);
    return success();
  }
};

void spirv::IAddCarryOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<IAddCarryFold>(context);
}

//===----------------------------------------------------------------------===//
// spirv.[S|U]MulExtended
//===----------------------------------------------------------------------===//

// We are required to use CompositeConstructOp to create a constant struct as
// they are not yet implemented as constant, hence we can not do so in a fold.
template <typename MulOp, bool IsSigned>
struct MulExtendedFold final : OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();
    Type constituentType = lhs.getType();

    // [su]mulextended (x, 0) = <0, 0>
    if (matchPattern(rhs, m_Zero())) {
      Value zero = spirv::ConstantOp::getZero(constituentType, loc, rewriter);
      Value constituents[2] = {zero, zero};
      rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, op.getType(),
                                                               constituents);
      return success();
    }

    // According to the SPIR-V spec:
    //
    // Result Type must be from OpTypeStruct.  The struct must have two
    // members...
    //
    // Member 0 of the result gets the low-order bits of the multiplication.
    //
    // Member 1 of the result gets the high-order bits of the multiplication.
    Attribute lhsAttr;
    Attribute rhsAttr;
    if (!matchPattern(lhs, m_Constant(&lhsAttr)) ||
        !matchPattern(rhs, m_Constant(&rhsAttr)))
      return failure();

    auto lowBits = constFoldBinaryOp<IntegerAttr>(
        {lhsAttr, rhsAttr},
        [](const APInt &a, const APInt &b) { return a * b; });

    if (!lowBits)
      return failure();

    auto highBits = constFoldBinaryOp<IntegerAttr>(
        {lhsAttr, rhsAttr}, [](const APInt &a, const APInt &b) {
          if (IsSigned) {
            return llvm::APIntOps::mulhs(a, b);
          } else {
            return llvm::APIntOps::mulhu(a, b);
          }
        });

    if (!highBits)
      return failure();

    Value lowBitsVal =
        spirv::ConstantOp::create(rewriter, loc, constituentType, lowBits);

    Value highBitsVal =
        spirv::ConstantOp::create(rewriter, loc, constituentType, highBits);

    // Create empty struct
    Value undef = spirv::UndefOp::create(rewriter, loc, op.getType());
    // Fill in lowBits at id 0
    Value intermediate =
        spirv::CompositeInsertOp::create(rewriter, loc, lowBitsVal, undef, 0);
    // Fill in highBits at id 1
    rewriter.replaceOpWithNewOp<spirv::CompositeInsertOp>(op, highBitsVal,
                                                          intermediate, 1);
    return success();
  }
};

using SMulExtendedOpFold = MulExtendedFold<spirv::SMulExtendedOp, true>;
void spirv::SMulExtendedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<SMulExtendedOpFold>(context);
}

struct UMulExtendedOpXOne final : OpRewritePattern<spirv::UMulExtendedOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::UMulExtendedOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getOperand1();
    Value rhs = op.getOperand2();
    Type constituentType = lhs.getType();

    // umulextended (x, 1) = <x, 0>
    if (matchPattern(rhs, m_One())) {
      Value zero = spirv::ConstantOp::getZero(constituentType, loc, rewriter);
      Value constituents[2] = {lhs, zero};
      rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, op.getType(),
                                                               constituents);
      return success();
    }

    return failure();
  }
};

using UMulExtendedOpFold = MulExtendedFold<spirv::UMulExtendedOp, false>;
void spirv::UMulExtendedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<UMulExtendedOpFold, UMulExtendedOpXOne>(context);
}

//===----------------------------------------------------------------------===//
// spirv.UMod
//===----------------------------------------------------------------------===//

// Input:
//    %0 = spirv.UMod %arg0, %const32 : i32
//    %1 = spirv.UMod %0, %const4 : i32
// Output:
//    %0 = spirv.UMod %arg0, %const32 : i32
//    %1 = spirv.UMod %arg0, %const4 : i32

// The transformation is only applied if one divisor is a multiple of the other.

struct UModSimplification final : OpRewritePattern<spirv::UModOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::UModOp umodOp,
                                PatternRewriter &rewriter) const override {
    auto prevUMod = umodOp.getOperand(0).getDefiningOp<spirv::UModOp>();
    if (!prevUMod)
      return failure();

    TypedAttr prevValue;
    TypedAttr currValue;
    if (!matchPattern(prevUMod.getOperand(1), m_Constant(&prevValue)) ||
        !matchPattern(umodOp.getOperand(1), m_Constant(&currValue)))
      return failure();

    // Ensure that previous divisor is a multiple of the current divisor. If
    // not, fail the transformation.
    bool isApplicable = false;
    if (auto prevInt = dyn_cast<IntegerAttr>(prevValue)) {
      auto currInt = cast<IntegerAttr>(currValue);
      isApplicable = prevInt.getValue().urem(currInt.getValue()) == 0;
    } else if (auto prevVec = dyn_cast<DenseElementsAttr>(prevValue)) {
      auto currVec = cast<DenseElementsAttr>(currValue);
      isApplicable = llvm::all_of(llvm::zip_equal(prevVec.getValues<APInt>(),
                                                  currVec.getValues<APInt>()),
                                  [](const auto &pair) {
                                    auto &[prev, curr] = pair;
                                    return prev.urem(curr) == 0;
                                  });
    }

    if (!isApplicable)
      return failure();

    // The transformation is safe. Replace the existing UMod operation with a
    // new UMod operation, using the original dividend and the current divisor.
    rewriter.replaceOpWithNewOp<spirv::UModOp>(
        umodOp, umodOp.getType(), prevUMod.getOperand(0), umodOp.getOperand(1));

    return success();
  }
};

void spirv::UModOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.insert<UModSimplification>(context);
}

//===----------------------------------------------------------------------===//
// spirv.BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::BitcastOp::fold(FoldAdaptor /*adaptor*/) {
  Value curInput = getOperand();
  if (getType() == curInput.getType())
    return curInput;

  // Look through nested bitcasts.
  if (auto prevCast = curInput.getDefiningOp<spirv::BitcastOp>()) {
    Value prevInput = prevCast.getOperand();
    if (prevInput.getType() == getType())
      return prevInput;

    getOperandMutable().assign(prevInput);
    return getResult();
  }

  // TODO(kuhar): Consider constant-folding the operand attribute.
  return {};
}

//===----------------------------------------------------------------------===//
// spirv.CompositeExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::CompositeExtractOp::fold(FoldAdaptor adaptor) {
  Value compositeOp = getComposite();

  while (auto insertOp =
             compositeOp.getDefiningOp<spirv::CompositeInsertOp>()) {
    if (getIndices() == insertOp.getIndices())
      return insertOp.getObject();
    compositeOp = insertOp.getComposite();
  }

  if (auto constructOp =
          compositeOp.getDefiningOp<spirv::CompositeConstructOp>()) {
    auto type = llvm::cast<spirv::CompositeType>(constructOp.getType());
    if (getIndices().size() == 1 &&
        constructOp.getConstituents().size() == type.getNumElements()) {
      auto i = llvm::cast<IntegerAttr>(*getIndices().begin());
      if (i.getValue().getSExtValue() <
          static_cast<int64_t>(constructOp.getConstituents().size()))
        return constructOp.getConstituents()[i.getValue().getSExtValue()];
    }
  }

  auto indexVector = llvm::map_to_vector(getIndices(), [](Attribute attr) {
    return static_cast<unsigned>(llvm::cast<IntegerAttr>(attr).getInt());
  });
  return extractCompositeElement(adaptor.getComposite(), indexVector);
}

//===----------------------------------------------------------------------===//
// spirv.Constant
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// spirv.IAdd
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IAddOp::fold(FoldAdaptor adaptor) {
  // x + 0 = x
  if (matchPattern(getOperand2(), m_Zero()))
    return getOperand1();

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) + b; });
}

//===----------------------------------------------------------------------===//
// spirv.IMul
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IMulOp::fold(FoldAdaptor adaptor) {
  // x * 0 == 0
  if (matchPattern(getOperand2(), m_Zero()))
    return getOperand2();
  // x * 1 = x
  if (matchPattern(getOperand2(), m_One()))
    return getOperand1();

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a, const APInt &b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// spirv.ISub
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ISubOp::fold(FoldAdaptor adaptor) {
  // x - x = 0
  if (getOperand1() == getOperand2())
    return Builder(getContext()).getZeroAttr(getType());

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) - b; });
}

//===----------------------------------------------------------------------===//
// spirv.SDiv
//===----------------------------------------------------------------------===//

OpFoldResult spirv::SDivOp::fold(FoldAdaptor adaptor) {
  // sdiv (x, 1) = x
  if (matchPattern(getOperand2(), m_One()))
    return getOperand1();

  // According to the SPIR-V spec:
  //
  // Signed-integer division of Operand 1 divided by Operand 2.
  // Results are computed per component. Behavior is undefined if Operand 2 is
  // 0. Behavior is undefined if Operand 2 is -1 and Operand 1 is the minimum
  // representable value for the operands' type, causing signed overflow.
  //
  // So don't fold during undefined behavior.
  bool div0OrOverflow = false;
  auto res = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        if (div0OrOverflow || isDivZeroOrOverflow(a, b)) {
          div0OrOverflow = true;
          return a;
        }
        return a.sdiv(b);
      });
  return div0OrOverflow ? Attribute() : res;
}

//===----------------------------------------------------------------------===//
// spirv.SMod
//===----------------------------------------------------------------------===//

OpFoldResult spirv::SModOp::fold(FoldAdaptor adaptor) {
  // smod (x, 1) = 0
  if (matchPattern(getOperand2(), m_One()))
    return Builder(getContext()).getZeroAttr(getType());

  // According to SPIR-V spec:
  //
  // Signed remainder operation for the remainder whose sign matches the sign
  // of Operand 2. Behavior is undefined if Operand 2 is 0. Behavior is
  // undefined if Operand 2 is -1 and Operand 1 is the minimum representable
  // value for the operands' type, causing signed overflow. Otherwise, the
  // result is the remainder r of Operand 1 divided by Operand 2 where if
  // r ≠ 0, the sign of r is the same as the sign of Operand 2.
  //
  // So don't fold during undefined behavior
  bool div0OrOverflow = false;
  auto res = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        if (div0OrOverflow || isDivZeroOrOverflow(a, b)) {
          div0OrOverflow = true;
          return a;
        }
        APInt c = a.abs().urem(b.abs());
        if (c.isZero())
          return c;
        if (b.isNegative()) {
          APInt zero = APInt::getZero(c.getBitWidth());
          return a.isNegative() ? (zero - c) : (b + c);
        }
        return a.isNegative() ? (b - c) : c;
      });
  return div0OrOverflow ? Attribute() : res;
}

//===----------------------------------------------------------------------===//
// spirv.SRem
//===----------------------------------------------------------------------===//

OpFoldResult spirv::SRemOp::fold(FoldAdaptor adaptor) {
  // x % 1 = 0
  if (matchPattern(getOperand2(), m_One()))
    return Builder(getContext()).getZeroAttr(getType());

  // According to SPIR-V spec:
  //
  // Signed remainder operation for the remainder whose sign matches the sign
  // of Operand 1. Behavior is undefined if Operand 2 is 0. Behavior is
  // undefined if Operand 2 is -1 and Operand 1 is the minimum representable
  // value for the operands' type, causing signed overflow. Otherwise, the
  // result is the remainder r of Operand 1 divided by Operand 2 where if
  // r ≠ 0, the sign of r is the same as the sign of Operand 1.

  // Don't fold if it would do undefined behavior.
  bool div0OrOverflow = false;
  auto res = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](APInt a, const APInt &b) {
        if (div0OrOverflow || isDivZeroOrOverflow(a, b)) {
          div0OrOverflow = true;
          return a;
        }
        return a.srem(b);
      });
  return div0OrOverflow ? Attribute() : res;
}

//===----------------------------------------------------------------------===//
// spirv.UDiv
//===----------------------------------------------------------------------===//

OpFoldResult spirv::UDivOp::fold(FoldAdaptor adaptor) {
  // udiv (x, 1) = x
  if (matchPattern(getOperand2(), m_One()))
    return getOperand1();

  // According to the SPIR-V spec:
  //
  // Unsigned-integer division of Operand 1 divided by Operand 2. Behavior is
  // undefined if Operand 2 is 0.
  //
  // So don't fold during undefined behavior.
  bool div0 = false;
  auto res = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        if (div0 || b.isZero()) {
          div0 = true;
          return a;
        }
        return a.udiv(b);
      });
  return div0 ? Attribute() : res;
}

//===----------------------------------------------------------------------===//
// spirv.UMod
//===----------------------------------------------------------------------===//

OpFoldResult spirv::UModOp::fold(FoldAdaptor adaptor) {
  // umod (x, 1) = 0
  if (matchPattern(getOperand2(), m_One()))
    return Builder(getContext()).getZeroAttr(getType());

  // According to the SPIR-V spec:
  //
  // Unsigned modulo operation of Operand 1 modulo Operand 2. Behavior is
  // undefined if Operand 2 is 0.
  //
  // So don't fold during undefined behavior.
  bool div0 = false;
  auto res = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        if (div0 || b.isZero()) {
          div0 = true;
          return a;
        }
        return a.urem(b);
      });
  return div0 ? Attribute() : res;
}

//===----------------------------------------------------------------------===//
// spirv.SNegate
//===----------------------------------------------------------------------===//

OpFoldResult spirv::SNegateOp::fold(FoldAdaptor adaptor) {
  // -(-x) = 0 - (0 - x) = x
  auto op = getOperand();
  if (auto negateOp = op.getDefiningOp<spirv::SNegateOp>())
    return negateOp->getOperand(0);

  // According to the SPIR-V spec:
  //
  // Signed-integer subtract of Operand from zero.
  return constFoldUnaryOp<IntegerAttr>(
      adaptor.getOperands(), [](const APInt &a) {
        APInt zero = APInt::getZero(a.getBitWidth());
        return zero - a;
      });
}

//===----------------------------------------------------------------------===//
// spirv.NotOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::NotOp::fold(spirv::NotOp::FoldAdaptor adaptor) {
  // !(!x) = x
  auto op = getOperand();
  if (auto notOp = op.getDefiningOp<spirv::NotOp>())
    return notOp->getOperand(0);

  // According to the SPIR-V spec:
  //
  // Complement the bits of Operand.
  return constFoldUnaryOp<IntegerAttr>(adaptor.getOperands(), [&](APInt a) {
    a.flipAllBits();
    return a;
  });
}

//===----------------------------------------------------------------------===//
// spirv.LogicalAnd
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalAndOp::fold(FoldAdaptor adaptor) {
  if (std::optional<bool> rhs =
          getScalarOrSplatBoolAttr(adaptor.getOperand2())) {
    // x && true = x
    if (*rhs)
      return getOperand1();

    // x && false = false
    if (!*rhs)
      return adaptor.getOperand2();
  }

  return Attribute();
}

//===----------------------------------------------------------------------===//
// spirv.LogicalEqualOp
//===----------------------------------------------------------------------===//

OpFoldResult
spirv::LogicalEqualOp::fold(spirv::LogicalEqualOp::FoldAdaptor adaptor) {
  // x == x -> true
  if (getOperand1() == getOperand2()) {
    auto trueAttr = BoolAttr::get(getContext(), true);
    if (isa<IntegerType>(getType()))
      return trueAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, trueAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [](const APInt &a, const APInt &b) {
        return a == b ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.LogicalNotEqualOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalNotEqualOp::fold(FoldAdaptor adaptor) {
  if (std::optional<bool> rhs =
          getScalarOrSplatBoolAttr(adaptor.getOperand2())) {
    // x != false -> x
    if (!rhs.value())
      return getOperand1();
  }

  // x == x -> false
  if (getOperand1() == getOperand2()) {
    auto falseAttr = BoolAttr::get(getContext(), false);
    if (isa<IntegerType>(getType()))
      return falseAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, falseAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [](const APInt &a, const APInt &b) {
        return a == b ? APInt::getZero(1) : APInt::getAllOnes(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.LogicalNot
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalNotOp::fold(FoldAdaptor adaptor) {
  // !(!x) = x
  auto op = getOperand();
  if (auto notOp = op.getDefiningOp<spirv::LogicalNotOp>())
    return notOp->getOperand(0);

  // According to the SPIR-V spec:
  //
  // Complement the bits of Operand.
  return constFoldUnaryOp<IntegerAttr>(adaptor.getOperands(),
                                       [](const APInt &a) {
                                         APInt zero = APInt::getZero(1);
                                         return a == 1 ? zero : (zero + 1);
                                       });
}

void spirv::LogicalNotOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results
      .add<ConvertLogicalNotOfIEqual, ConvertLogicalNotOfINotEqual,
           ConvertLogicalNotOfLogicalEqual, ConvertLogicalNotOfLogicalNotEqual>(
          context);
}

//===----------------------------------------------------------------------===//
// spirv.LogicalOr
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalOrOp::fold(FoldAdaptor adaptor) {
  if (auto rhs = getScalarOrSplatBoolAttr(adaptor.getOperand2())) {
    if (*rhs) {
      // x || true = true
      return adaptor.getOperand2();
    }

    if (!*rhs) {
      // x || false = x
      return getOperand1();
    }
  }

  return Attribute();
}

//===----------------------------------------------------------------------===//
// spirv.SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::SelectOp::fold(FoldAdaptor adaptor) {
  // spirv.Select _ x x -> x
  Value trueVals = getTrueValue();
  Value falseVals = getFalseValue();
  if (trueVals == falseVals)
    return trueVals;

  ArrayRef<Attribute> operands = adaptor.getOperands();

  // spirv.Select true  x y -> x
  // spirv.Select false x y -> y
  if (auto boolAttr = getScalarOrSplatBoolAttr(operands[0]))
    return *boolAttr ? trueVals : falseVals;

  // Check that all the operands are constant
  if (!operands[0] || !operands[1] || !operands[2])
    return Attribute();

  // Note: getScalarOrSplatBoolAttr will always return a boolAttr if we are in
  // the scalar case. Hence, we are only required to consider the case of
  // DenseElementsAttr in foldSelectOp.
  auto condAttrs = dyn_cast<DenseElementsAttr>(operands[0]);
  auto trueAttrs = dyn_cast<DenseElementsAttr>(operands[1]);
  auto falseAttrs = dyn_cast<DenseElementsAttr>(operands[2]);
  if (!condAttrs || !trueAttrs || !falseAttrs)
    return Attribute();

  auto elementResults = llvm::to_vector<4>(trueAttrs.getValues<Attribute>());
  auto iters = llvm::zip_equal(elementResults, condAttrs.getValues<BoolAttr>(),
                               falseAttrs.getValues<Attribute>());
  for (auto [result, cond, falseRes] : iters) {
    if (!cond.getValue())
      result = falseRes;
  }

  auto resultType = trueAttrs.getType();
  return DenseElementsAttr::get(cast<ShapedType>(resultType), elementResults);
}

//===----------------------------------------------------------------------===//
// spirv.IEqualOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IEqualOp::fold(spirv::IEqualOp::FoldAdaptor adaptor) {
  // x == x -> true
  if (getOperand1() == getOperand2()) {
    auto trueAttr = BoolAttr::get(getContext(), true);
    if (isa<IntegerType>(getType()))
      return trueAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, trueAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a == b ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.INotEqualOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::INotEqualOp::fold(spirv::INotEqualOp::FoldAdaptor adaptor) {
  // x == x -> false
  if (getOperand1() == getOperand2()) {
    auto falseAttr = BoolAttr::get(getContext(), false);
    if (isa<IntegerType>(getType()))
      return falseAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, falseAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a == b ? APInt::getZero(1) : APInt::getAllOnes(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.SGreaterThan
//===----------------------------------------------------------------------===//

OpFoldResult
spirv::SGreaterThanOp::fold(spirv::SGreaterThanOp::FoldAdaptor adaptor) {
  // x == x -> false
  if (getOperand1() == getOperand2()) {
    auto falseAttr = BoolAttr::get(getContext(), false);
    if (isa<IntegerType>(getType()))
      return falseAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, falseAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a.sgt(b) ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.SGreaterThanEqual
//===----------------------------------------------------------------------===//

OpFoldResult spirv::SGreaterThanEqualOp::fold(
    spirv::SGreaterThanEqualOp::FoldAdaptor adaptor) {
  // x == x -> true
  if (getOperand1() == getOperand2()) {
    auto trueAttr = BoolAttr::get(getContext(), true);
    if (isa<IntegerType>(getType()))
      return trueAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, trueAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a.sge(b) ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.UGreaterThan
//===----------------------------------------------------------------------===//

OpFoldResult
spirv::UGreaterThanOp::fold(spirv::UGreaterThanOp::FoldAdaptor adaptor) {
  // x == x -> false
  if (getOperand1() == getOperand2()) {
    auto falseAttr = BoolAttr::get(getContext(), false);
    if (isa<IntegerType>(getType()))
      return falseAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, falseAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a.ugt(b) ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.UGreaterThanEqual
//===----------------------------------------------------------------------===//

OpFoldResult spirv::UGreaterThanEqualOp::fold(
    spirv::UGreaterThanEqualOp::FoldAdaptor adaptor) {
  // x == x -> true
  if (getOperand1() == getOperand2()) {
    auto trueAttr = BoolAttr::get(getContext(), true);
    if (isa<IntegerType>(getType()))
      return trueAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, trueAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a.uge(b) ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.SLessThan
//===----------------------------------------------------------------------===//

OpFoldResult spirv::SLessThanOp::fold(spirv::SLessThanOp::FoldAdaptor adaptor) {
  // x == x -> false
  if (getOperand1() == getOperand2()) {
    auto falseAttr = BoolAttr::get(getContext(), false);
    if (isa<IntegerType>(getType()))
      return falseAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, falseAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a.slt(b) ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.SLessThanEqual
//===----------------------------------------------------------------------===//

OpFoldResult
spirv::SLessThanEqualOp::fold(spirv::SLessThanEqualOp::FoldAdaptor adaptor) {
  // x == x -> true
  if (getOperand1() == getOperand2()) {
    auto trueAttr = BoolAttr::get(getContext(), true);
    if (isa<IntegerType>(getType()))
      return trueAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, trueAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a.sle(b) ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.ULessThan
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ULessThanOp::fold(spirv::ULessThanOp::FoldAdaptor adaptor) {
  // x == x -> false
  if (getOperand1() == getOperand2()) {
    auto falseAttr = BoolAttr::get(getContext(), false);
    if (isa<IntegerType>(getType()))
      return falseAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, falseAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a.ult(b) ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.ULessThanEqual
//===----------------------------------------------------------------------===//

OpFoldResult
spirv::ULessThanEqualOp::fold(spirv::ULessThanEqualOp::FoldAdaptor adaptor) {
  // x == x -> true
  if (getOperand1() == getOperand2()) {
    auto trueAttr = BoolAttr::get(getContext(), true);
    if (isa<IntegerType>(getType()))
      return trueAttr;
    if (auto vecTy = dyn_cast<VectorType>(getType()))
      return SplatElementsAttr::get(vecTy, trueAttr);
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getType(), [](const APInt &a, const APInt &b) {
        return a.ule(b) ? APInt::getAllOnes(1) : APInt::getZero(1);
      });
}

//===----------------------------------------------------------------------===//
// spirv.ShiftLeftLogical
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ShiftLeftLogicalOp::fold(
    spirv::ShiftLeftLogicalOp::FoldAdaptor adaptor) {
  // x << 0 -> x
  if (matchPattern(adaptor.getOperand2(), m_Zero())) {
    return getOperand1();
  }

  // Unfortunately due to below undefined behaviour can't fold 0 for Base.

  // Results are computed per component, and within each component, per bit...
  //
  // The result is undefined if Shift is greater than or equal to the bit width
  // of the components of Base.
  //
  // So we can use the APInt << method, but don't fold if undefined behaviour.
  bool shiftToLarge = false;
  auto res = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        if (shiftToLarge || b.uge(a.getBitWidth())) {
          shiftToLarge = true;
          return a;
        }
        return a << b;
      });
  return shiftToLarge ? Attribute() : res;
}

//===----------------------------------------------------------------------===//
// spirv.ShiftRightArithmetic
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ShiftRightArithmeticOp::fold(
    spirv::ShiftRightArithmeticOp::FoldAdaptor adaptor) {
  // x >> 0 -> x
  if (matchPattern(adaptor.getOperand2(), m_Zero())) {
    return getOperand1();
  }

  // Unfortunately due to below undefined behaviour can't fold 0, -1 for Base.

  // Results are computed per component, and within each component, per bit...
  //
  // The result is undefined if Shift is greater than or equal to the bit width
  // of the components of Base.
  //
  // So we can use the APInt ashr method, but don't fold if undefined behaviour.
  bool shiftToLarge = false;
  auto res = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        if (shiftToLarge || b.uge(a.getBitWidth())) {
          shiftToLarge = true;
          return a;
        }
        return a.ashr(b);
      });
  return shiftToLarge ? Attribute() : res;
}

//===----------------------------------------------------------------------===//
// spirv.ShiftRightLogical
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ShiftRightLogicalOp::fold(
    spirv::ShiftRightLogicalOp::FoldAdaptor adaptor) {
  // x >> 0 -> x
  if (matchPattern(adaptor.getOperand2(), m_Zero())) {
    return getOperand1();
  }

  // Unfortunately due to below undefined behaviour can't fold 0 for Base.

  // Results are computed per component, and within each component, per bit...
  //
  // The result is undefined if Shift is greater than or equal to the bit width
  // of the components of Base.
  //
  // So we can use the APInt lshr method, but don't fold if undefined behaviour.
  bool shiftToLarge = false;
  auto res = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        if (shiftToLarge || b.uge(a.getBitWidth())) {
          shiftToLarge = true;
          return a;
        }
        return a.lshr(b);
      });
  return shiftToLarge ? Attribute() : res;
}

//===----------------------------------------------------------------------===//
// spirv.BitwiseAndOp
//===----------------------------------------------------------------------===//

OpFoldResult
spirv::BitwiseAndOp::fold(spirv::BitwiseAndOp::FoldAdaptor adaptor) {
  // x & x -> x
  if (getOperand1() == getOperand2()) {
    return getOperand1();
  }

  APInt rhsMask;
  if (matchPattern(adaptor.getOperand2(), m_ConstantInt(&rhsMask))) {
    // x & 0 -> 0
    if (rhsMask.isZero())
      return getOperand2();

    // x & <all ones> -> x
    if (rhsMask.isAllOnes())
      return getOperand1();

    // (UConvert x : iN to iK) & <mask with N low bits set> -> UConvert x
    if (auto zext = getOperand1().getDefiningOp<spirv::UConvertOp>()) {
      int valueBits =
          getElementTypeOrSelf(zext.getOperand()).getIntOrFloatBitWidth();
      if (rhsMask.zextOrTrunc(valueBits).isAllOnes())
        return getOperand1();
    }
  }

  // According to the SPIR-V spec:
  //
  // Type is a scalar or vector of integer type.
  // Results are computed per component, and within each component, per bit.
  // So we can use the APInt & method.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a, const APInt &b) { return a & b; });
}

//===----------------------------------------------------------------------===//
// spirv.BitwiseOrOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::BitwiseOrOp::fold(spirv::BitwiseOrOp::FoldAdaptor adaptor) {
  // x | x -> x
  if (getOperand1() == getOperand2()) {
    return getOperand1();
  }

  APInt rhsMask;
  if (matchPattern(adaptor.getOperand2(), m_ConstantInt(&rhsMask))) {
    // x | 0 -> x
    if (rhsMask.isZero())
      return getOperand1();

    // x | <all ones> -> <all ones>
    if (rhsMask.isAllOnes())
      return getOperand2();
  }

  // According to the SPIR-V spec:
  //
  // Type is a scalar or vector of integer type.
  // Results are computed per component, and within each component, per bit.
  // So we can use the APInt | method.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a, const APInt &b) { return a | b; });
}

//===----------------------------------------------------------------------===//
// spirv.BitwiseXorOp
//===----------------------------------------------------------------------===//

OpFoldResult
spirv::BitwiseXorOp::fold(spirv::BitwiseXorOp::FoldAdaptor adaptor) {
  // x ^ 0 -> x
  if (matchPattern(adaptor.getOperand2(), m_Zero())) {
    return getOperand1();
  }

  // x ^ x -> 0
  if (getOperand1() == getOperand2())
    return Builder(getContext()).getZeroAttr(getType());

  // According to the SPIR-V spec:
  //
  // Type is a scalar or vector of integer type.
  // Results are computed per component, and within each component, per bit.
  // So we can use the APInt ^ method.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a, const APInt &b) { return a ^ b; });
}

//===----------------------------------------------------------------------===//
// spirv.mlir.selection
//===----------------------------------------------------------------------===//

namespace {
// Blocks from the given `spirv.mlir.selection` operation must satisfy the
// following layout:
//
//       +-----------------------------------------------+
//       | header block                                  |
//       | spirv.BranchConditionalOp %cond, ^case0, ^case1 |
//       +-----------------------------------------------+
//                            /   \
//                             ...
//
//
//   +------------------------+    +------------------------+
//   | case #0                |    | case #1                |
//   | spirv.Store %ptr %value0 |    | spirv.Store %ptr %value1 |
//   | spirv.Branch ^merge      |    | spirv.Branch ^merge      |
//   +------------------------+    +------------------------+
//
//
//                             ...
//                            \   /
//                              v
//                       +-------------+
//                       | merge block |
//                       +-------------+
//
struct ConvertSelectionOpToSelect final : OpRewritePattern<spirv::SelectionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::SelectionOp selectionOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = selectionOp.getOperation();
    Region &body = op->getRegion(0);
    // Verifier allows an empty region for `spirv.mlir.selection`.
    if (body.empty()) {
      return failure();
    }

    // Check that region consists of 4 blocks:
    // header block, `true` block, `false` block and merge block.
    if (llvm::range_size(body) != 4) {
      return failure();
    }

    Block *headerBlock = selectionOp.getHeaderBlock();
    if (!onlyContainsBranchConditionalOp(headerBlock)) {
      return failure();
    }

    auto brConditionalOp =
        cast<spirv::BranchConditionalOp>(headerBlock->front());

    Block *trueBlock = brConditionalOp.getSuccessor(0);
    Block *falseBlock = brConditionalOp.getSuccessor(1);
    Block *mergeBlock = selectionOp.getMergeBlock();

    if (failed(canCanonicalizeSelection(trueBlock, falseBlock, mergeBlock)))
      return failure();

    Value trueValue = getSrcValue(trueBlock);
    Value falseValue = getSrcValue(falseBlock);
    Value ptrValue = getDstPtr(trueBlock);
    auto storeOpAttributes =
        cast<spirv::StoreOp>(trueBlock->front())->getAttrs();

    auto selectOp = spirv::SelectOp::create(
        rewriter, selectionOp.getLoc(), trueValue.getType(),
        brConditionalOp.getCondition(), trueValue, falseValue);
    spirv::StoreOp::create(rewriter, selectOp.getLoc(), ptrValue,
                           selectOp.getResult(), storeOpAttributes);

    // `spirv.mlir.selection` is not needed anymore.
    rewriter.eraseOp(op);
    return success();
  }

private:
  // Checks that given blocks follow the following rules:
  // 1. Each conditional block consists of two operations, the first operation
  //    is a `spirv.Store` and the last operation is a `spirv.Branch`.
  // 2. Each `spirv.Store` uses the same pointer and the same memory attributes.
  // 3. A control flow goes into the given merge block from the given
  //    conditional blocks.
  LogicalResult canCanonicalizeSelection(Block *trueBlock, Block *falseBlock,
                                         Block *mergeBlock) const;

  bool onlyContainsBranchConditionalOp(Block *block) const {
    return llvm::hasSingleElement(*block) &&
           isa<spirv::BranchConditionalOp>(block->front());
  }

  bool isSameAttrList(spirv::StoreOp lhs, spirv::StoreOp rhs) const {
    return lhs->getDiscardableAttrDictionary() ==
               rhs->getDiscardableAttrDictionary() &&
           lhs.getProperties() == rhs.getProperties();
  }

  // Returns a source value for the given block.
  Value getSrcValue(Block *block) const {
    auto storeOp = cast<spirv::StoreOp>(block->front());
    return storeOp.getValue();
  }

  // Returns a destination value for the given block.
  Value getDstPtr(Block *block) const {
    auto storeOp = cast<spirv::StoreOp>(block->front());
    return storeOp.getPtr();
  }
};

LogicalResult ConvertSelectionOpToSelect::canCanonicalizeSelection(
    Block *trueBlock, Block *falseBlock, Block *mergeBlock) const {
  // Each block must consists of 2 operations.
  if (llvm::range_size(*trueBlock) != 2 || llvm::range_size(*falseBlock) != 2) {
    return failure();
  }

  auto trueBrStoreOp = dyn_cast<spirv::StoreOp>(trueBlock->front());
  auto trueBrBranchOp =
      dyn_cast<spirv::BranchOp>(*std::next(trueBlock->begin()));
  auto falseBrStoreOp = dyn_cast<spirv::StoreOp>(falseBlock->front());
  auto falseBrBranchOp =
      dyn_cast<spirv::BranchOp>(*std::next(falseBlock->begin()));

  if (!trueBrStoreOp || !trueBrBranchOp || !falseBrStoreOp ||
      !falseBrBranchOp) {
    return failure();
  }

  // Checks that given type is valid for `spirv.SelectOp`.
  // According to SPIR-V spec:
  // "Before version 1.4, Result Type must be a pointer, scalar, or vector.
  // Starting with version 1.4, Result Type can additionally be a composite type
  // other than a vector."
  bool isScalarOrVector =
      llvm::cast<spirv::SPIRVType>(trueBrStoreOp.getValue().getType())
          .isScalarOrVector();

  // Check that each `spirv.Store` uses the same pointer, memory access
  // attributes and a valid type of the value.
  if ((trueBrStoreOp.getPtr() != falseBrStoreOp.getPtr()) ||
      !isSameAttrList(trueBrStoreOp, falseBrStoreOp) || !isScalarOrVector) {
    return failure();
  }

  if ((trueBrBranchOp->getSuccessor(0) != mergeBlock) ||
      (falseBrBranchOp->getSuccessor(0) != mergeBlock)) {
    return failure();
  }

  return success();
}
} // namespace

void spirv::SelectionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<ConvertSelectionOpToSelect>(context);
}
