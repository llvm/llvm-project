//===- ArithmeticOps.cpp - MLIR Arithmetic dialect ops implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Pattern helpers
//===----------------------------------------------------------------------===//

static IntegerAttr addIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return builder.getIntegerAttr(res.getType(),
                                lhs.cast<IntegerAttr>().getInt() +
                                    rhs.cast<IntegerAttr>().getInt());
}

static IntegerAttr subIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return builder.getIntegerAttr(res.getType(),
                                lhs.cast<IntegerAttr>().getInt() -
                                    rhs.cast<IntegerAttr>().getInt());
}

/// Invert an integer comparison predicate.
static arith::CmpIPredicate invertPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return arith::CmpIPredicate::ne;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::eq;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ule;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ult;
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

static arith::CmpIPredicateAttr invertPredicate(arith::CmpIPredicateAttr pred) {
  return arith::CmpIPredicateAttr::get(pred.getContext(),
                                       invertPredicate(pred.getValue()));
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "ArithmeticCanonicalization.inc"
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AddIOp::fold(ArrayRef<Attribute> operands) {
  // addi(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a + b; });
}

void arith::AddIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<AddIAddConstant, AddISubConstantRHS, AddISubConstantLHS>(
      context);
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::SubIOp::fold(ArrayRef<Attribute> operands) {
  // subi(x,x) -> 0
  if (getOperand(0) == getOperand(1))
    return Builder(getContext()).getZeroAttr(getType());
  // subi(x,0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a - b; });
}

void arith::SubIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<SubIRHSAddConstant, SubILHSAddConstant, SubIRHSSubConstantRHS,
                  SubIRHSSubConstantLHS, SubILHSSubConstantRHS,
                  SubILHSSubConstantLHS>(context);
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MulIOp::fold(ArrayRef<Attribute> operands) {
  // muli(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();
  // muli(x, 1) -> x
  if (matchPattern(rhs(), m_One()))
    return getOperand(0);
  // TODO: Handle the overflow case.

  // default folder
  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivUIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (div0 || !b) {
      div0 = true;
      return a;
    }
    return a.udiv(b);
  });

  // Fold out division by one. Assumes all tensors of all ones are splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return lhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return lhs();
  }

  return div0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivSIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    return a.sdiv_ov(b, overflowOrDiv0);
  });

  // Fold out division by one. Assumes all tensors of all ones are splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return lhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return lhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// Ceil and floor division folding helpers
//===----------------------------------------------------------------------===//

static APInt signedCeilNonnegInputs(APInt a, APInt b, bool &overflow) {
  // Returns (a-1)/b + 1
  APInt one(a.getBitWidth(), 1, true); // Signed value 1.
  APInt val = a.ssub_ov(one, overflow).sdiv_ov(b, overflow);
  return val.sadd_ov(one, overflow);
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::CeilDivSIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    unsigned bits = a.getBitWidth();
    APInt zero = APInt::getZero(bits);
    if (a.sgt(zero) && b.sgt(zero)) {
      // Both positive, return ceil(a, b).
      return signedCeilNonnegInputs(a, b, overflowOrDiv0);
    }
    if (a.slt(zero) && b.slt(zero)) {
      // Both negative, return ceil(-a, -b).
      APInt posA = zero.ssub_ov(a, overflowOrDiv0);
      APInt posB = zero.ssub_ov(b, overflowOrDiv0);
      return signedCeilNonnegInputs(posA, posB, overflowOrDiv0);
    }
    if (a.slt(zero) && b.sgt(zero)) {
      // A is negative, b is positive, return - ( -a / b).
      APInt posA = zero.ssub_ov(a, overflowOrDiv0);
      APInt div = posA.sdiv_ov(b, overflowOrDiv0);
      return zero.ssub_ov(div, overflowOrDiv0);
    }
    // A is positive (or zero), b is negative, return - (a / -b).
    APInt posB = zero.ssub_ov(b, overflowOrDiv0);
    APInt div = a.sdiv_ov(posB, overflowOrDiv0);
    return zero.ssub_ov(div, overflowOrDiv0);
  });

  // Fold out floor division by one. Assumes all tensors of all ones are
  // splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return lhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return lhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::FloorDivSIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, APInt b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    unsigned bits = a.getBitWidth();
    APInt zero = APInt::getZero(bits);
    if (a.sge(zero) && b.sgt(zero)) {
      // Both positive (or a is zero), return a / b.
      return a.sdiv_ov(b, overflowOrDiv0);
    }
    if (a.sle(zero) && b.slt(zero)) {
      // Both negative (or a is zero), return -a / -b.
      APInt posA = zero.ssub_ov(a, overflowOrDiv0);
      APInt posB = zero.ssub_ov(b, overflowOrDiv0);
      return posA.sdiv_ov(posB, overflowOrDiv0);
    }
    if (a.slt(zero) && b.sgt(zero)) {
      // A is negative, b is positive, return - ceil(-a, b).
      APInt posA = zero.ssub_ov(a, overflowOrDiv0);
      APInt ceil = signedCeilNonnegInputs(posA, b, overflowOrDiv0);
      return zero.ssub_ov(ceil, overflowOrDiv0);
    }
    // A is positive, b is negative, return - ceil(a, -b).
    APInt posB = zero.ssub_ov(b, overflowOrDiv0);
    APInt ceil = signedCeilNonnegInputs(a, posB, overflowOrDiv0);
    return zero.ssub_ov(ceil, overflowOrDiv0);
  });

  // Fold out floor division by one. Assumes all tensors of all ones are
  // splats.
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue() == 1)
      return lhs();
  } else if (auto rhs = operands[1].dyn_cast_or_null<SplatElementsAttr>()) {
    if (rhs.getSplatValue<IntegerAttr>().getValue() == 1)
      return lhs();
  }

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemUIOp::fold(ArrayRef<Attribute> operands) {
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  return IntegerAttr::get(lhs.getType(), lhs.getValue().urem(rhsValue));
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemSIOp::fold(ArrayRef<Attribute> operands) {
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!rhs)
    return {};
  auto rhsValue = rhs.getValue();

  // x % 1 = 0
  if (rhsValue.isOneValue())
    return IntegerAttr::get(rhs.getType(), APInt(rhsValue.getBitWidth(), 0));

  // Don't fold if it requires division by zero.
  if (rhsValue.isNullValue())
    return {};

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};
  return IntegerAttr::get(lhs.getType(), lhs.getValue().srem(rhsValue));
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AndIOp::fold(ArrayRef<Attribute> operands) {
  /// and(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();
  /// and(x, allOnes) -> x
  APInt intValue;
  if (matchPattern(rhs(), m_ConstantInt(&intValue)) && intValue.isAllOnes())
    return lhs();
  /// and(x, x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::OrIOp::fold(ArrayRef<Attribute> operands) {
  /// or(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();
  /// or(x, x) -> x
  if (lhs() == rhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a | b; });
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::XOrIOp::fold(ArrayRef<Attribute> operands) {
  /// xor(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();
  /// xor(x, x) -> 0
  if (lhs() == rhs())
    return Builder(getContext()).getZeroAttr(getType());

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a ^ b; });
}

void arith::XOrIOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<XOrINotCmpI>(context);
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AddFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::SubFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MulFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(
      operands, [](APFloat a, APFloat b) { return a / b; });
}

//===----------------------------------------------------------------------===//
// Verifiers for integer and floating point extension/truncation ops
//===----------------------------------------------------------------------===//

// Extend ops can only extend to a wider type.
template <typename ValType, typename Op>
static LogicalResult verifyExtOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.in().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() >= dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

// Truncate ops can only truncate to a shorter type.
template <typename ValType, typename Op>
static LogicalResult verifyTruncateOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.in().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() <= dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be shorter than operand type " << srcType;

  return success();
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ExtUIOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>())
    return IntegerAttr::get(
        getType(), lhs.getValue().zext(getType().getIntOrFloatBitWidth()));

  return {};
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ExtSIOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>())
    return IntegerAttr::get(
        getType(), lhs.getValue().sext(getType().getIntOrFloatBitWidth()));

  return {};
}

// TODO temporary fixes until second patch is in
OpFoldResult arith::TruncFOp::fold(ArrayRef<Attribute> operands) {
  return {};
}

bool arith::TruncFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

OpFoldResult arith::TruncIOp::fold(ArrayRef<Attribute> operands) {
  return {};
}

bool arith::TruncIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

bool arith::ExtUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

bool arith::ExtSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

bool arith::ExtFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

OpFoldResult arith::ConstantOp::fold(ArrayRef<Attribute> operands) {
  return {};
}

bool arith::SIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

bool arith::UIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

bool arith::FPToSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

bool arith::FPToUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return true;
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

bool arith::IndexCastOp::areCastCompatible(TypeRange inputs,
                                           TypeRange outputs) {
  assert(inputs.size() == 1 && outputs.size() == 1 &&
         "index_cast op expects one result and one result");

  // Shape equivalence is guaranteed by op traits.
  auto srcType = getElementTypeOrSelf(inputs.front());
  auto dstType = getElementTypeOrSelf(outputs.front());

  return (srcType.isIndex() && dstType.isSignlessInteger()) ||
         (srcType.isSignlessInteger() && dstType.isIndex());
}

OpFoldResult arith::IndexCastOp::fold(ArrayRef<Attribute> operands) {
  // index_cast(constant) -> constant
  // A little hack because we go through int. Otherwise, the size of the
  // constant might need to change.
  if (auto value = operands[0].dyn_cast_or_null<IntegerAttr>())
    return IntegerAttr::get(getType(), value.getInt());

  return {};
}

void arith::IndexCastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<IndexCastOfIndexCast, IndexCastOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

bool arith::BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  assert(inputs.size() == 1 && outputs.size() == 1 &&
         "bitcast op expects one operand and one result");

  // Shape equivalence is guaranteed by op traits.
  auto srcType = getElementTypeOrSelf(inputs.front());
  auto dstType = getElementTypeOrSelf(outputs.front());

  // Types are guarnateed to be integers or floats by constraints.
  return srcType.getIntOrFloatBitWidth() == dstType.getIntOrFloatBitWidth();
}

OpFoldResult arith::BitcastOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "bitcast op expects 1 operand");

  auto resType = getType();
  auto operand = operands[0];
  if (!operand)
    return {};

  /// Bitcast dense elements.
  if (auto denseAttr = operand.dyn_cast_or_null<DenseElementsAttr>())
    return denseAttr.bitcast(resType.cast<ShapedType>().getElementType());
  /// Other shaped types unhandled.
  if (resType.isa<ShapedType>())
    return {};

  /// Bitcast integer or float to integer or float.
  APInt bits = operand.isa<FloatAttr>()
                   ? operand.cast<FloatAttr>().getValue().bitcastToAPInt()
                   : operand.cast<IntegerAttr>().getValue();

  if (auto resFloatType = resType.dyn_cast<FloatType>())
    return FloatAttr::get(resType,
                          APFloat(resFloatType.getFloatSemantics(), bits));
  return IntegerAttr::get(resType, bits);
}

void arith::BitcastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<BitcastOfBitcast>(context);
}

//===----------------------------------------------------------------------===//
// Helpers for compare ops
//===----------------------------------------------------------------------===//

/// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), i1Type);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
/// comparison predicates.
bool mlir::arith::applyCmpPredicate(arith::CmpIPredicate predicate,
                                    const APInt &lhs, const APInt &rhs) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return lhs.eq(rhs);
  case arith::CmpIPredicate::ne:
    return lhs.ne(rhs);
  case arith::CmpIPredicate::slt:
    return lhs.slt(rhs);
  case arith::CmpIPredicate::sle:
    return lhs.sle(rhs);
  case arith::CmpIPredicate::sgt:
    return lhs.sgt(rhs);
  case arith::CmpIPredicate::sge:
    return lhs.sge(rhs);
  case arith::CmpIPredicate::ult:
    return lhs.ult(rhs);
  case arith::CmpIPredicate::ule:
    return lhs.ule(rhs);
  case arith::CmpIPredicate::ugt:
    return lhs.ugt(rhs);
  case arith::CmpIPredicate::uge:
    return lhs.uge(rhs);
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

/// Returns true if the predicate is true for two equal operands.
static bool applyCmpPredicateToEqualOperands(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::uge:
    return true;
  case arith::CmpIPredicate::ne:
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ugt:
    return false;
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

OpFoldResult arith::CmpIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpi takes two operands");

  // cmpi(pred, x, x)
  if (lhs() == rhs()) {
    auto val = applyCmpPredicateToEqualOperands(getPredicate());
    return BoolAttr::get(getContext(), val);
  }

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  auto rhs = operands.back().dyn_cast_or_null<IntegerAttr>();
  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
bool mlir::arith::applyCmpPredicate(arith::CmpFPredicate predicate,
                                    const APFloat &lhs, const APFloat &rhs) {
  auto cmpResult = lhs.compare(rhs);
  switch (predicate) {
  case arith::CmpFPredicate::AlwaysFalse:
    return false;
  case arith::CmpFPredicate::OEQ:
    return cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::OGT:
    return cmpResult == APFloat::cmpGreaterThan;
  case arith::CmpFPredicate::OGE:
    return cmpResult == APFloat::cmpGreaterThan ||
           cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::OLT:
    return cmpResult == APFloat::cmpLessThan;
  case arith::CmpFPredicate::OLE:
    return cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::ONE:
    return cmpResult != APFloat::cmpUnordered && cmpResult != APFloat::cmpEqual;
  case arith::CmpFPredicate::ORD:
    return cmpResult != APFloat::cmpUnordered;
  case arith::CmpFPredicate::UEQ:
    return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::UGT:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpGreaterThan;
  case arith::CmpFPredicate::UGE:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpGreaterThan ||
           cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::ULT:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpLessThan;
  case arith::CmpFPredicate::ULE:
    return cmpResult == APFloat::cmpUnordered ||
           cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
  case arith::CmpFPredicate::UNE:
    return cmpResult != APFloat::cmpEqual;
  case arith::CmpFPredicate::UNO:
    return cmpResult == APFloat::cmpUnordered;
  case arith::CmpFPredicate::AlwaysTrue:
    return true;
  }
  llvm_unreachable("unknown cmpf predicate kind");
}

OpFoldResult arith::CmpFOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpf takes two operands");

  auto lhs = operands.front().dyn_cast_or_null<FloatAttr>();
  auto rhs = operands.back().dyn_cast_or_null<FloatAttr>();

  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/ArithmeticOpsEnums.cpp.inc"
