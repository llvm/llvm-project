//===- ArithmeticOps.cpp - MLIR Arithmetic dialect ops implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"

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
arith::CmpIPredicate arith::invertPredicate(arith::CmpIPredicate pred) {
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
} // namespace

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void arith::ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto type = getType();
  if (auto intCst = getValue().dyn_cast<IntegerAttr>()) {
    auto intType = type.dyn_cast<IntegerType>();

    // Sugar i1 constants with 'true' and 'false'.
    if (intType && intType.getWidth() == 1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, build a complex name with the value and type.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getValue();
    if (intType)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
  } else {
    setNameFn(getResult(), "cst");
  }
}

/// TODO: disallow arith.constant to return anything other than signless integer
/// or float like.
LogicalResult arith::ConstantOp::verify() {
  auto type = getType();
  // The value's type must match the return type.
  if (getValue().getType() != type) {
    return emitOpError() << "value type " << getValue().getType()
                         << " must match return type: " << type;
  }
  // Integer values must be signless.
  if (type.isa<IntegerType>() && !type.cast<IntegerType>().isSignless())
    return emitOpError("integer return type must be signless");
  // Any float or elements attribute are acceptable.
  if (!getValue().isa<IntegerAttr, FloatAttr, ElementsAttr>()) {
    return emitOpError(
        "value must be an integer, float, or elements attribute");
  }
  return success();
}

bool arith::ConstantOp::isBuildableWith(Attribute value, Type type) {
  // The value's type must be the same as the provided type.
  auto typedAttr = value.dyn_cast<TypedAttr>();
  if (!typedAttr || typedAttr.getType() != type)
    return false;
  // Integer values must be signless.
  if (type.isa<IntegerType>() && !type.cast<IntegerType>().isSignless())
    return false;
  // Integer, float, and element attributes are buildable.
  return value.isa<IntegerAttr, FloatAttr, ElementsAttr>();
}

OpFoldResult arith::ConstantOp::fold(ArrayRef<Attribute> operands) {
  return getValue();
}

void arith::ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                                 int64_t value, unsigned width) {
  auto type = builder.getIntegerType(width);
  arith::ConstantOp::build(builder, result, type,
                           builder.getIntegerAttr(type, value));
}

void arith::ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                                 int64_t value, Type type) {
  assert(type.isSignlessInteger() &&
         "ConstantIntOp can only have signless integer type values");
  arith::ConstantOp::build(builder, result, type,
                           builder.getIntegerAttr(type, value));
}

bool arith::ConstantIntOp::classof(Operation *op) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op))
    return constOp.getType().isSignlessInteger();
  return false;
}

void arith::ConstantFloatOp::build(OpBuilder &builder, OperationState &result,
                                   const APFloat &value, FloatType type) {
  arith::ConstantOp::build(builder, result, type,
                           builder.getFloatAttr(type, value));
}

bool arith::ConstantFloatOp::classof(Operation *op) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op))
    return constOp.getType().isa<FloatType>();
  return false;
}

void arith::ConstantIndexOp::build(OpBuilder &builder, OperationState &result,
                                   int64_t value) {
  arith::ConstantOp::build(builder, result, builder.getIndexType(),
                           builder.getIndexAttr(value));
}

bool arith::ConstantIndexOp::classof(Operation *op) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op))
    return constOp.getType().isIndex();
  return false;
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AddIOp::fold(ArrayRef<Attribute> operands) {
  // addi(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  // addi(subi(a, b), b) -> a
  if (auto sub = getLhs().getDefiningOp<SubIOp>())
    if (getRhs() == sub.getRhs())
      return sub.getLhs();

  // addi(b, subi(a, b)) -> a
  if (auto sub = getRhs().getDefiningOp<SubIOp>())
    if (getLhs() == sub.getRhs())
      return sub.getLhs();

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) + b; });
}

void arith::AddIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<AddIAddConstant, AddISubConstantRHS, AddISubConstantLHS>(
      context);
}

//===----------------------------------------------------------------------===//
// AddUICarryOp
//===----------------------------------------------------------------------===//

Optional<SmallVector<int64_t, 4>> arith::AddUICarryOp::getShapeForUnroll() {
  if (auto vt = getType(0).dyn_cast<VectorType>())
    return llvm::to_vector<4>(vt.getShape());
  return None;
}

// Returns the carry bit, assuming that `sum` is the result of addition of
// `operand` and another number.
static APInt calculateCarry(const APInt &sum, const APInt &operand) {
  return sum.ult(operand) ? APInt::getAllOnes(1) : APInt::getZero(1);
}

LogicalResult
arith::AddUICarryOp::fold(ArrayRef<Attribute> operands,
                          SmallVectorImpl<OpFoldResult> &results) {
  auto carryTy = getCarry().getType();
  // addui_carry(x, 0) -> x, false
  if (matchPattern(getRhs(), m_Zero())) {
    auto carryZero = APInt::getZero(1);
    Builder builder(getContext());
    auto falseValue = builder.getZeroAttr(carryTy);

    results.push_back(getLhs());
    results.push_back(falseValue);
    return success();
  }

  // addui_carry(constant_a, constant_b) -> constant_sum, constant_carry
  // Let the `constFoldBinaryOp` utility attempt to fold the sum of both
  // operands. If that succeeds, calculate the carry boolean based on the sum
  // and the first (constant) operand, `lhs`. Note that we cannot simply call
  // `constFoldBinaryOp` again to calculate the carry (bit) because the
  // constructed attribute is of the same element type as both operands.
  if (Attribute sumAttr = constFoldBinaryOp<IntegerAttr>(
          operands, [](APInt a, const APInt &b) { return std::move(a) + b; })) {
    Attribute carryAttr;
    if (auto lhs = operands[0].dyn_cast<IntegerAttr>()) {
      // Both arguments are scalars, calculate the scalar carry value.
      auto sum = sumAttr.cast<IntegerAttr>();
      carryAttr = IntegerAttr::get(
          carryTy, calculateCarry(sum.getValue(), lhs.getValue()));
    } else if (auto lhs = operands[0].dyn_cast<SplatElementsAttr>()) {
      // Both arguments are splats, calculate the splat carry value.
      auto sum = sumAttr.cast<SplatElementsAttr>();
      APInt carry = calculateCarry(sum.getSplatValue<APInt>(),
                                   lhs.getSplatValue<APInt>());
      carryAttr = SplatElementsAttr::get(carryTy, carry);
    } else if (auto lhs = operands[0].dyn_cast<ElementsAttr>()) {
      // Othwerwise calculate element-wise carry values.
      auto sum = sumAttr.cast<ElementsAttr>();
      const auto numElems = static_cast<size_t>(sum.getNumElements());
      SmallVector<APInt> carryValues;
      carryValues.reserve(numElems);

      auto sumIt = sum.value_begin<APInt>();
      auto lhsIt = lhs.value_begin<APInt>();
      for (size_t i = 0, e = numElems; i != e; ++i, ++sumIt, ++lhsIt)
        carryValues.push_back(calculateCarry(*sumIt, *lhsIt));

      carryAttr = DenseElementsAttr::get(carryTy, carryValues);
    } else {
      return failure();
    }

    results.push_back(sumAttr);
    results.push_back(carryAttr);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::SubIOp::fold(ArrayRef<Attribute> operands) {
  // subi(x,x) -> 0
  if (getOperand(0) == getOperand(1))
    return Builder(getContext()).getZeroAttr(getType());
  // subi(x,0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) - b; });
}

void arith::SubIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns
      .add<SubIRHSAddConstant, SubILHSAddConstant, SubIRHSSubConstantRHS,
           SubIRHSSubConstantLHS, SubILHSSubConstantRHS, SubILHSSubConstantLHS>(
          context);
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MulIOp::fold(ArrayRef<Attribute> operands) {
  // muli(x, 0) -> 0
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs();
  // muli(x, 1) -> x
  if (matchPattern(getRhs(), m_One()))
    return getOperand(0);
  // TODO: Handle the overflow case.

  // default folder
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](const APInt &a, const APInt &b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivUIOp::fold(ArrayRef<Attribute> operands) {
  // divui (x, 1) -> x.
  if (matchPattern(getRhs(), m_One()))
    return getLhs();

  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (div0 || !b) {
          div0 = true;
          return a;
        }
        return a.udiv(b);
      });

  return div0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivSIOp::fold(ArrayRef<Attribute> operands) {
  // divsi (x, 1) -> x.
  if (matchPattern(getRhs(), m_One()))
    return getLhs();

  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        return a.sdiv_ov(b, overflowOrDiv0);
      });

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// Ceil and floor division folding helpers
//===----------------------------------------------------------------------===//

static APInt signedCeilNonnegInputs(const APInt &a, const APInt &b,
                                    bool &overflow) {
  // Returns (a-1)/b + 1
  APInt one(a.getBitWidth(), 1, true); // Signed value 1.
  APInt val = a.ssub_ov(one, overflow).sdiv_ov(b, overflow);
  return val.sadd_ov(one, overflow);
}

//===----------------------------------------------------------------------===//
// CeilDivUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::CeilDivUIOp::fold(ArrayRef<Attribute> operands) {
  // ceildivui (x, 1) -> x.
  if (matchPattern(getRhs(), m_One()))
    return getLhs();

  bool overflowOrDiv0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        APInt quotient = a.udiv(b);
        if (!a.urem(b))
          return quotient;
        APInt one(a.getBitWidth(), 1, true);
        return quotient.uadd_ov(one, overflowOrDiv0);
      });

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::CeilDivSIOp::fold(ArrayRef<Attribute> operands) {
  // ceildivsi (x, 1) -> x.
  if (matchPattern(getRhs(), m_One()))
    return getLhs();

  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        if (!a)
          return a;
        // After this point we know that neither a or b are zero.
        unsigned bits = a.getBitWidth();
        APInt zero = APInt::getZero(bits);
        bool aGtZero = a.sgt(zero);
        bool bGtZero = b.sgt(zero);
        if (aGtZero && bGtZero) {
          // Both positive, return ceil(a, b).
          return signedCeilNonnegInputs(a, b, overflowOrDiv0);
        }
        if (!aGtZero && !bGtZero) {
          // Both negative, return ceil(-a, -b).
          APInt posA = zero.ssub_ov(a, overflowOrDiv0);
          APInt posB = zero.ssub_ov(b, overflowOrDiv0);
          return signedCeilNonnegInputs(posA, posB, overflowOrDiv0);
        }
        if (!aGtZero && bGtZero) {
          // A is negative, b is positive, return - ( -a / b).
          APInt posA = zero.ssub_ov(a, overflowOrDiv0);
          APInt div = posA.sdiv_ov(b, overflowOrDiv0);
          return zero.ssub_ov(div, overflowOrDiv0);
        }
        // A is positive, b is negative, return - (a / -b).
        APInt posB = zero.ssub_ov(b, overflowOrDiv0);
        APInt div = a.sdiv_ov(posB, overflowOrDiv0);
        return zero.ssub_ov(div, overflowOrDiv0);
      });

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::FloorDivSIOp::fold(ArrayRef<Attribute> operands) {
  // floordivsi (x, 1) -> x.
  if (matchPattern(getRhs(), m_One()))
    return getLhs();

  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        if (!a)
          return a;
        // After this point we know that neither a or b are zero.
        unsigned bits = a.getBitWidth();
        APInt zero = APInt::getZero(bits);
        bool aGtZero = a.sgt(zero);
        bool bGtZero = b.sgt(zero);
        if (aGtZero && bGtZero) {
          // Both positive, return a / b.
          return a.sdiv_ov(b, overflowOrDiv0);
        }
        if (!aGtZero && !bGtZero) {
          // Both negative, return -a / -b.
          APInt posA = zero.ssub_ov(a, overflowOrDiv0);
          APInt posB = zero.ssub_ov(b, overflowOrDiv0);
          return posA.sdiv_ov(posB, overflowOrDiv0);
        }
        if (!aGtZero && bGtZero) {
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

  return overflowOrDiv0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemUIOp::fold(ArrayRef<Attribute> operands) {
  // remui (x, 1) -> 0.
  if (matchPattern(getRhs(), m_One()))
    return Builder(getContext()).getZeroAttr(getType());

  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (div0 || b.isNullValue()) {
          div0 = true;
          return a;
        }
        return a.urem(b);
      });

  return div0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemSIOp::fold(ArrayRef<Attribute> operands) {
  // remsi (x, 1) -> 0.
  if (matchPattern(getRhs(), m_One()))
    return Builder(getContext()).getZeroAttr(getType());

  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result =
      constFoldBinaryOp<IntegerAttr>(operands, [&](APInt a, const APInt &b) {
        if (div0 || b.isNullValue()) {
          div0 = true;
          return a;
        }
        return a.srem(b);
      });

  return div0 ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AndIOp::fold(ArrayRef<Attribute> operands) {
  /// and(x, 0) -> 0
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs();
  /// and(x, allOnes) -> x
  APInt intValue;
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isAllOnes())
    return getLhs();
  /// and(x, not(x)) -> 0
  if (matchPattern(getRhs(), m_Op<XOrIOp>(matchers::m_Val(getLhs()),
                                          m_ConstantInt(&intValue))) &&
      intValue.isAllOnes())
    return IntegerAttr::get(getType(), 0);
  /// and(not(x), x) -> 0
  if (matchPattern(getLhs(), m_Op<XOrIOp>(matchers::m_Val(getRhs()),
                                          m_ConstantInt(&intValue))) &&
      intValue.isAllOnes())
    return IntegerAttr::get(getType(), 0);

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) & b; });
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::OrIOp::fold(ArrayRef<Attribute> operands) {
  /// or(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  /// or(x, <all ones>) -> <all ones>
  if (auto rhsAttr = operands[1].dyn_cast_or_null<IntegerAttr>())
    if (rhsAttr.getValue().isAllOnes())
      return rhsAttr;

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) | b; });
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::XOrIOp::fold(ArrayRef<Attribute> operands) {
  /// xor(x, 0) -> x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  /// xor(x, x) -> 0
  if (getLhs() == getRhs())
    return Builder(getContext()).getZeroAttr(getType());
  /// xor(xor(x, a), a) -> x
  if (arith::XOrIOp prev = getLhs().getDefiningOp<arith::XOrIOp>())
    if (prev.getRhs() == getRhs())
      return prev.getLhs();

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, const APInt &b) { return std::move(a) ^ b; });
}

void arith::XOrIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<XOrINotCmpI>(context);
}

//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::NegFOp::fold(ArrayRef<Attribute> operands) {
  /// negf(negf(x)) -> x
  if (auto op = this->getOperand().getDefiningOp<arith::NegFOp>())
    return op.getOperand();
  return constFoldUnaryOp<FloatAttr>(operands,
                                     [](const APFloat &a) { return -a; });
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AddFOp::fold(ArrayRef<Attribute> operands) {
  // addf(x, -0) -> x
  if (matchPattern(getRhs(), m_NegZeroFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::SubFOp::fold(ArrayRef<Attribute> operands) {
  // subf(x, +0) -> x
  if (matchPattern(getRhs(), m_PosZeroFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// MaxFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MaxFOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "maxf takes two operands");

  // maxf(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  // maxf(x, -inf) -> x
  if (matchPattern(getRhs(), m_NegInfFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      operands,
      [](const APFloat &a, const APFloat &b) { return llvm::maximum(a, b); });
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

OpFoldResult MaxSIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // maxsi(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  APInt intValue;
  // maxsi(x,MAX_INT) -> MAX_INT
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isMaxSignedValue())
    return getRhs();

  // maxsi(x, MIN_INT) -> x
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isMinSignedValue())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::smax(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

OpFoldResult MaxUIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // maxui(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  APInt intValue;
  // maxui(x,MAX_INT) -> MAX_INT
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isMaxValue())
    return getRhs();

  // maxui(x, MIN_INT) -> x
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isMinValue())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::umax(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MinFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MinFOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "minf takes two operands");

  // minf(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  // minf(x, +inf) -> x
  if (matchPattern(getRhs(), m_PosInfFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      operands,
      [](const APFloat &a, const APFloat &b) { return llvm::minimum(a, b); });
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

OpFoldResult MinSIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // minsi(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  APInt intValue;
  // minsi(x,MIN_INT) -> MIN_INT
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isMinSignedValue())
    return getRhs();

  // minsi(x, MAX_INT) -> x
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isMaxSignedValue())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::smin(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

OpFoldResult MinUIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "binary operation takes two operands");

  // minui(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  APInt intValue;
  // minui(x,MIN_INT) -> MIN_INT
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isMinValue())
    return getRhs();

  // minui(x, MAX_INT) -> x
  if (matchPattern(getRhs(), m_ConstantInt(&intValue)) && intValue.isMaxValue())
    return getLhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::umin(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MulFOp::fold(ArrayRef<Attribute> operands) {
  // mulf(x, 1) -> x
  if (matchPattern(getRhs(), m_OneFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) { return a * b; });
}

void arith::MulFOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<MulFOfNegF>(context);
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivFOp::fold(ArrayRef<Attribute> operands) {
  // divf(x, 1) -> x
  if (matchPattern(getRhs(), m_OneFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      operands, [](const APFloat &a, const APFloat &b) { return a / b; });
}

void arith::DivFOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<DivFOfNegF>(context);
}

//===----------------------------------------------------------------------===//
// RemFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemFOp::fold(ArrayRef<Attribute> operands) {
  return constFoldBinaryOp<FloatAttr>(operands,
                                      [](const APFloat &a, const APFloat &b) {
                                        APFloat result(a);
                                        (void)result.remainder(b);
                                        return result;
                                      });
}

//===----------------------------------------------------------------------===//
// Utility functions for verifying cast ops
//===----------------------------------------------------------------------===//

template <typename... Types>
using type_list = std::tuple<Types...> *;

/// Returns a non-null type only if the provided type is one of the allowed
/// types or one of the allowed shaped types of the allowed types. Returns the
/// element type if a valid shaped type is provided.
template <typename... ShapedTypes, typename... ElementTypes>
static Type getUnderlyingType(Type type, type_list<ShapedTypes...>,
                              type_list<ElementTypes...>) {
  if (type.isa<ShapedType>() && !type.isa<ShapedTypes...>())
    return {};

  auto underlyingType = getElementTypeOrSelf(type);
  if (!underlyingType.isa<ElementTypes...>())
    return {};

  return underlyingType;
}

/// Get allowed underlying types for vectors and tensors.
template <typename... ElementTypes>
static Type getTypeIfLike(Type type) {
  return getUnderlyingType(type, type_list<VectorType, TensorType>(),
                           type_list<ElementTypes...>());
}

/// Get allowed underlying types for vectors, tensors, and memrefs.
template <typename... ElementTypes>
static Type getTypeIfLikeOrMemRef(Type type) {
  return getUnderlyingType(type,
                           type_list<VectorType, TensorType, MemRefType>(),
                           type_list<ElementTypes...>());
}

static bool areValidCastInputsAndOutputs(TypeRange inputs, TypeRange outputs) {
  return inputs.size() == 1 && outputs.size() == 1 &&
         succeeded(verifyCompatibleShapes(inputs.front(), outputs.front()));
}

//===----------------------------------------------------------------------===//
// Verifiers for integer and floating point extension/truncation ops
//===----------------------------------------------------------------------===//

// Extend ops can only extend to a wider type.
template <typename ValType, typename Op>
static LogicalResult verifyExtOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.getIn().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() >= dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

// Truncate ops can only truncate to a shorter type.
template <typename ValType, typename Op>
static LogicalResult verifyTruncateOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.getIn().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() <= dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be shorter than operand type " << srcType;

  return success();
}

/// Validate a cast that changes the width of a type.
template <template <typename> class WidthComparator, typename... ElementTypes>
static bool checkWidthChangeCast(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLike<ElementTypes...>(inputs.front());
  auto dstType = getTypeIfLike<ElementTypes...>(outputs.front());
  if (!srcType || !dstType)
    return false;

  return WidthComparator<unsigned>()(dstType.getIntOrFloatBitWidth(),
                                     srcType.getIntOrFloatBitWidth());
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ExtUIOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = getIn().getDefiningOp<ExtUIOp>()) {
    getInMutable().assign(lhs.getIn());
    return getResult();
  }
  Type resType = getType();
  unsigned bitWidth;
  if (auto shapedType = resType.dyn_cast<ShapedType>())
    bitWidth = shapedType.getElementTypeBitWidth();
  else
    bitWidth = resType.getIntOrFloatBitWidth();
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      operands, getType(), [bitWidth](const APInt &a, bool &castStatus) {
        return a.zext(bitWidth);
      });
}

bool arith::ExtUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::greater, IntegerType>(inputs, outputs);
}

LogicalResult arith::ExtUIOp::verify() {
  return verifyExtOp<IntegerType>(*this);
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ExtSIOp::fold(ArrayRef<Attribute> operands) {
  if (auto lhs = getIn().getDefiningOp<ExtSIOp>()) {
    getInMutable().assign(lhs.getIn());
    return getResult();
  }
  Type resType = getType();
  unsigned bitWidth;
  if (auto shapedType = resType.dyn_cast<ShapedType>())
    bitWidth = shapedType.getElementTypeBitWidth();
  else
    bitWidth = resType.getIntOrFloatBitWidth();
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      operands, getType(), [bitWidth](const APInt &a, bool &castStatus) {
        return a.sext(bitWidth);
      });
}

bool arith::ExtSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::greater, IntegerType>(inputs, outputs);
}

void arith::ExtSIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add<ExtSIOfExtUI>(context);
}

LogicalResult arith::ExtSIOp::verify() {
  return verifyExtOp<IntegerType>(*this);
}

//===----------------------------------------------------------------------===//
// ExtFOp
//===----------------------------------------------------------------------===//

bool arith::ExtFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::greater, FloatType>(inputs, outputs);
}

LogicalResult arith::ExtFOp::verify() { return verifyExtOp<FloatType>(*this); }

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::TruncIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary operation takes one operand");

  // trunci(zexti(a)) -> a
  // trunci(sexti(a)) -> a
  if (matchPattern(getOperand(), m_Op<arith::ExtUIOp>()) ||
      matchPattern(getOperand(), m_Op<arith::ExtSIOp>()))
    return getOperand().getDefiningOp()->getOperand(0);

  // trunci(trunci(a)) -> trunci(a))
  if (matchPattern(getOperand(), m_Op<arith::TruncIOp>())) {
    setOperand(getOperand().getDefiningOp()->getOperand(0));
    return getResult();
  }

  Type resType = getType();
  unsigned bitWidth;
  if (auto shapedType = resType.dyn_cast<ShapedType>())
    bitWidth = shapedType.getElementTypeBitWidth();
  else
    bitWidth = resType.getIntOrFloatBitWidth();

  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      operands, getType(), [bitWidth](const APInt &a, bool &castStatus) {
        return a.trunc(bitWidth);
      });
}

bool arith::TruncIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::less, IntegerType>(inputs, outputs);
}

LogicalResult arith::TruncIOp::verify() {
  return verifyTruncateOp<IntegerType>(*this);
}

//===----------------------------------------------------------------------===//
// TruncFOp
//===----------------------------------------------------------------------===//

/// Perform safe const propagation for truncf, i.e. only propagate if FP value
/// can be represented without precision loss or rounding.
OpFoldResult arith::TruncFOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "unary operation takes one operand");

  auto constOperand = operands.front();
  if (!constOperand || !constOperand.isa<FloatAttr>())
    return {};

  // Convert to target type via 'double'.
  double sourceValue =
      constOperand.dyn_cast<FloatAttr>().getValue().convertToDouble();
  auto targetAttr = FloatAttr::get(getType(), sourceValue);

  // Propagate if constant's value does not change after truncation.
  if (sourceValue == targetAttr.getValue().convertToDouble())
    return targetAttr;

  return {};
}

bool arith::TruncFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::less, FloatType>(inputs, outputs);
}

LogicalResult arith::TruncFOp::verify() {
  return verifyTruncateOp<FloatType>(*this);
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

void arith::AndIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<AndOfExtUI, AndOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

void arith::OrIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add<OrOfExtUI, OrOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// Verifiers for casts between integers and floats.
//===----------------------------------------------------------------------===//

template <typename From, typename To>
static bool checkIntFloatCast(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLike<From>(inputs.front());
  auto dstType = getTypeIfLike<To>(outputs.back());

  return srcType && dstType;
}

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

bool arith::UIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkIntFloatCast<IntegerType, FloatType>(inputs, outputs);
}

OpFoldResult arith::UIToFPOp::fold(ArrayRef<Attribute> operands) {
  Type resType = getType();
  Type resEleType;
  if (auto shapedType = resType.dyn_cast<ShapedType>())
    resEleType = shapedType.getElementType();
  else
    resEleType = resType;
  return constFoldCastOp<IntegerAttr, FloatAttr>(
      operands, getType(), [&resEleType](const APInt &a, bool &castStatus) {
        FloatType floatTy = resEleType.cast<FloatType>();
        APFloat apf(floatTy.getFloatSemantics(),
                    APInt::getZero(floatTy.getWidth()));
        apf.convertFromAPInt(a, /*IsSigned=*/false,
                             APFloat::rmNearestTiesToEven);
        return apf;
      });
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

bool arith::SIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkIntFloatCast<IntegerType, FloatType>(inputs, outputs);
}

OpFoldResult arith::SIToFPOp::fold(ArrayRef<Attribute> operands) {
  Type resType = getType();
  Type resEleType;
  if (auto shapedType = resType.dyn_cast<ShapedType>())
    resEleType = shapedType.getElementType();
  else
    resEleType = resType;
  return constFoldCastOp<IntegerAttr, FloatAttr>(
      operands, getType(), [&resEleType](const APInt &a, bool &castStatus) {
        FloatType floatTy = resEleType.cast<FloatType>();
        APFloat apf(floatTy.getFloatSemantics(),
                    APInt::getZero(floatTy.getWidth()));
        apf.convertFromAPInt(a, /*IsSigned=*/true,
                             APFloat::rmNearestTiesToEven);
        return apf;
      });
}
//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

bool arith::FPToUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkIntFloatCast<FloatType, IntegerType>(inputs, outputs);
}

OpFoldResult arith::FPToUIOp::fold(ArrayRef<Attribute> operands) {
  Type resType = getType();
  Type resEleType;
  if (auto shapedType = resType.dyn_cast<ShapedType>())
    resEleType = shapedType.getElementType();
  else
    resEleType = resType;
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      operands, getType(), [&resEleType](const APFloat &a, bool &castStatus) {
        IntegerType intTy = resEleType.cast<IntegerType>();
        bool ignored;
        APSInt api(intTy.getWidth(), /*isUnsigned=*/true);
        castStatus = APFloat::opInvalidOp !=
                     a.convertToInteger(api, APFloat::rmTowardZero, &ignored);
        return api;
      });
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

bool arith::FPToSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkIntFloatCast<FloatType, IntegerType>(inputs, outputs);
}

OpFoldResult arith::FPToSIOp::fold(ArrayRef<Attribute> operands) {
  Type resType = getType();
  Type resEleType;
  if (auto shapedType = resType.dyn_cast<ShapedType>())
    resEleType = shapedType.getElementType();
  else
    resEleType = resType;
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      operands, getType(), [&resEleType](const APFloat &a, bool &castStatus) {
        IntegerType intTy = resEleType.cast<IntegerType>();
        bool ignored;
        APSInt api(intTy.getWidth(), /*isUnsigned=*/false);
        castStatus = APFloat::opInvalidOp !=
                     a.convertToInteger(api, APFloat::rmTowardZero, &ignored);
        return api;
      });
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

bool arith::IndexCastOp::areCastCompatible(TypeRange inputs,
                                           TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(outputs.front());
  if (!srcType || !dstType)
    return false;

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
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<IndexCastOfIndexCast, IndexCastOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

bool arith::BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType =
      getTypeIfLikeOrMemRef<IntegerType, IndexType, FloatType>(inputs.front());
  auto dstType =
      getTypeIfLikeOrMemRef<IntegerType, IndexType, FloatType>(outputs.front());
  if (!srcType || !dstType)
    return false;

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

void arith::BitcastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<BitcastOfBitcast>(context);
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
    return VectorType::get(vectorType.getShape(), i1Type,
                           vectorType.getNumScalableDims());
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

static Attribute getBoolAttribute(Type type, MLIRContext *ctx, bool value) {
  auto boolAttr = BoolAttr::get(ctx, value);
  ShapedType shapedType = type.dyn_cast_or_null<ShapedType>();
  if (!shapedType)
    return boolAttr;
  return DenseElementsAttr::get(shapedType, boolAttr);
}

OpFoldResult arith::CmpIOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2 && "cmpi takes two operands");

  // cmpi(pred, x, x)
  if (getLhs() == getRhs()) {
    auto val = applyCmpPredicateToEqualOperands(getPredicate());
    return getBoolAttribute(getType(), getContext(), val);
  }

  if (matchPattern(getRhs(), m_Zero())) {
    if (auto extOp = getLhs().getDefiningOp<ExtSIOp>()) {
      // extsi(%x : i1 -> iN) != 0  ->  %x
      if (extOp.getOperand().getType().cast<IntegerType>().getWidth() == 1 &&
          getPredicate() == arith::CmpIPredicate::ne)
        return extOp.getOperand();
    }
    if (auto extOp = getLhs().getDefiningOp<ExtUIOp>()) {
      // extui(%x : i1 -> iN) != 0  ->  %x
      if (extOp.getOperand().getType().cast<IntegerType>().getWidth() == 1 &&
          getPredicate() == arith::CmpIPredicate::ne)
        return extOp.getOperand();
    }
  }

  // Move constant to the right side.
  if (operands[0] && !operands[1]) {
    // Do not use invertPredicate, as it will change eq to ne and vice versa.
    using Pred = CmpIPredicate;
    const std::pair<Pred, Pred> invPreds[] = {
        {Pred::slt, Pred::sgt}, {Pred::sgt, Pred::slt}, {Pred::sle, Pred::sge},
        {Pred::sge, Pred::sle}, {Pred::ult, Pred::ugt}, {Pred::ugt, Pred::ult},
        {Pred::ule, Pred::uge}, {Pred::uge, Pred::ule}, {Pred::eq, Pred::eq},
        {Pred::ne, Pred::ne},
    };
    Pred origPred = getPredicate();
    for (auto pred : invPreds) {
      if (origPred == pred.first) {
        setPredicateAttr(CmpIPredicateAttr::get(getContext(), pred.second));
        Value lhs = getLhs();
        Value rhs = getRhs();
        getLhsMutable().assign(rhs);
        getRhsMutable().assign(lhs);
        return getResult();
      }
    }
    llvm_unreachable("unknown cmpi predicate kind");
  }

  auto lhs = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!lhs)
    return {};

  // We are moving constants to the right side; So if lhs is constant rhs is
  // guaranteed to be a constant.
  auto rhs = operands.back().cast<IntegerAttr>();

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

void arith::CmpIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.insert<CmpIExtSI, CmpIExtUI>(context);
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

  // If one operand is NaN, making them both NaN does not change the result.
  if (lhs && lhs.getValue().isNaN())
    rhs = lhs;
  if (rhs && rhs.getValue().isNaN())
    lhs = rhs;

  if (!lhs || !rhs)
    return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

class CmpFIntToFPConst final : public OpRewritePattern<CmpFOp> {
public:
  using OpRewritePattern<CmpFOp>::OpRewritePattern;

  static CmpIPredicate convertToIntegerPredicate(CmpFPredicate pred,
                                                 bool isUnsigned) {
    using namespace arith;
    switch (pred) {
    case CmpFPredicate::UEQ:
    case CmpFPredicate::OEQ:
      return CmpIPredicate::eq;
    case CmpFPredicate::UGT:
    case CmpFPredicate::OGT:
      return isUnsigned ? CmpIPredicate::ugt : CmpIPredicate::sgt;
    case CmpFPredicate::UGE:
    case CmpFPredicate::OGE:
      return isUnsigned ? CmpIPredicate::uge : CmpIPredicate::sge;
    case CmpFPredicate::ULT:
    case CmpFPredicate::OLT:
      return isUnsigned ? CmpIPredicate::ult : CmpIPredicate::slt;
    case CmpFPredicate::ULE:
    case CmpFPredicate::OLE:
      return isUnsigned ? CmpIPredicate::ule : CmpIPredicate::sle;
    case CmpFPredicate::UNE:
    case CmpFPredicate::ONE:
      return CmpIPredicate::ne;
    default:
      llvm_unreachable("Unexpected predicate!");
    }
  }

  LogicalResult matchAndRewrite(CmpFOp op,
                                PatternRewriter &rewriter) const override {
    FloatAttr flt;
    if (!matchPattern(op.getRhs(), m_Constant(&flt)))
      return failure();

    const APFloat &rhs = flt.getValue();

    // Don't attempt to fold a nan.
    if (rhs.isNaN())
      return failure();

    // Get the width of the mantissa.  We don't want to hack on conversions that
    // might lose information from the integer, e.g. "i64 -> float"
    FloatType floatTy = op.getRhs().getType().cast<FloatType>();
    int mantissaWidth = floatTy.getFPMantissaWidth();
    if (mantissaWidth <= 0)
      return failure();

    bool isUnsigned;
    Value intVal;

    if (auto si = op.getLhs().getDefiningOp<SIToFPOp>()) {
      isUnsigned = false;
      intVal = si.getIn();
    } else if (auto ui = op.getLhs().getDefiningOp<UIToFPOp>()) {
      isUnsigned = true;
      intVal = ui.getIn();
    } else {
      return failure();
    }

    // Check to see that the input is converted from an integer type that is
    // small enough that preserves all bits.
    auto intTy = intVal.getType().cast<IntegerType>();
    auto intWidth = intTy.getWidth();

    // Number of bits representing values, as opposed to the sign
    auto valueBits = isUnsigned ? intWidth : (intWidth - 1);

    // Following test does NOT adjust intWidth downwards for signed inputs,
    // because the most negative value still requires all the mantissa bits
    // to distinguish it from one less than that value.
    if ((int)intWidth > mantissaWidth) {
      // Conversion would lose accuracy. Check if loss can impact comparison.
      int exponent = ilogb(rhs);
      if (exponent == APFloat::IEK_Inf) {
        int maxExponent = ilogb(APFloat::getLargest(rhs.getSemantics()));
        if (maxExponent < (int)valueBits) {
          // Conversion could create infinity.
          return failure();
        }
      } else {
        // Note that if rhs is zero or NaN, then Exp is negative
        // and first condition is trivially false.
        if (mantissaWidth <= exponent && exponent <= (int)valueBits) {
          // Conversion could affect comparison.
          return failure();
        }
      }
    }

    // Convert to equivalent cmpi predicate
    CmpIPredicate pred;
    switch (op.getPredicate()) {
    case CmpFPredicate::ORD:
      // Int to fp conversion doesn't create a nan (ord checks neither is a nan)
      rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/true,
                                                 /*width=*/1);
      return success();
    case CmpFPredicate::UNO:
      // Int to fp conversion doesn't create a nan (uno checks either is a nan)
      rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/false,
                                                 /*width=*/1);
      return success();
    default:
      pred = convertToIntegerPredicate(op.getPredicate(), isUnsigned);
      break;
    }

    if (!isUnsigned) {
      // If the rhs value is > SignedMax, fold the comparison.  This handles
      // +INF and large values.
      APFloat signedMax(rhs.getSemantics());
      signedMax.convertFromAPInt(APInt::getSignedMaxValue(intWidth), true,
                                 APFloat::rmNearestTiesToEven);
      if (signedMax < rhs) { // smax < 13123.0
        if (pred == CmpIPredicate::ne || pred == CmpIPredicate::slt ||
            pred == CmpIPredicate::sle)
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/true,
                                                     /*width=*/1);
        else
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/false,
                                                     /*width=*/1);
        return success();
      }
    } else {
      // If the rhs value is > UnsignedMax, fold the comparison. This handles
      // +INF and large values.
      APFloat unsignedMax(rhs.getSemantics());
      unsignedMax.convertFromAPInt(APInt::getMaxValue(intWidth), false,
                                   APFloat::rmNearestTiesToEven);
      if (unsignedMax < rhs) { // umax < 13123.0
        if (pred == CmpIPredicate::ne || pred == CmpIPredicate::ult ||
            pred == CmpIPredicate::ule)
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/true,
                                                     /*width=*/1);
        else
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/false,
                                                     /*width=*/1);
        return success();
      }
    }

    if (!isUnsigned) {
      // See if the rhs value is < SignedMin.
      APFloat signedMin(rhs.getSemantics());
      signedMin.convertFromAPInt(APInt::getSignedMinValue(intWidth), true,
                                 APFloat::rmNearestTiesToEven);
      if (signedMin > rhs) { // smin > 12312.0
        if (pred == CmpIPredicate::ne || pred == CmpIPredicate::sgt ||
            pred == CmpIPredicate::sge)
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/true,
                                                     /*width=*/1);
        else
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/false,
                                                     /*width=*/1);
        return success();
      }
    } else {
      // See if the rhs value is < UnsignedMin.
      APFloat unsignedMin(rhs.getSemantics());
      unsignedMin.convertFromAPInt(APInt::getMinValue(intWidth), false,
                                   APFloat::rmNearestTiesToEven);
      if (unsignedMin > rhs) { // umin > 12312.0
        if (pred == CmpIPredicate::ne || pred == CmpIPredicate::ugt ||
            pred == CmpIPredicate::uge)
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/true,
                                                     /*width=*/1);
        else
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/false,
                                                     /*width=*/1);
        return success();
      }
    }

    // Okay, now we know that the FP constant fits in the range [SMIN, SMAX] or
    // [0, UMAX], but it may still be fractional.  See if it is fractional by
    // casting the FP value to the integer value and back, checking for
    // equality. Don't do this for zero, because -0.0 is not fractional.
    bool ignored;
    APSInt rhsInt(intWidth, isUnsigned);
    if (APFloat::opInvalidOp ==
        rhs.convertToInteger(rhsInt, APFloat::rmTowardZero, &ignored)) {
      // Undefined behavior invoked - the destination type can't represent
      // the input constant.
      return failure();
    }

    if (!rhs.isZero()) {
      APFloat apf(floatTy.getFloatSemantics(),
                  APInt::getZero(floatTy.getWidth()));
      apf.convertFromAPInt(rhsInt, !isUnsigned, APFloat::rmNearestTiesToEven);

      bool equal = apf == rhs;
      if (!equal) {
        // If we had a comparison against a fractional value, we have to adjust
        // the compare predicate and sometimes the value.  rhsInt is rounded
        // towards zero at this point.
        switch (pred) {
        case CmpIPredicate::ne: // (float)int != 4.4   --> true
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/true,
                                                     /*width=*/1);
          return success();
        case CmpIPredicate::eq: // (float)int == 4.4   --> false
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/false,
                                                     /*width=*/1);
          return success();
        case CmpIPredicate::ule:
          // (float)int <= 4.4   --> int <= 4
          // (float)int <= -4.4  --> false
          if (rhs.isNegative()) {
            rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/false,
                                                       /*width=*/1);
            return success();
          }
          break;
        case CmpIPredicate::sle:
          // (float)int <= 4.4   --> int <= 4
          // (float)int <= -4.4  --> int < -4
          if (rhs.isNegative())
            pred = CmpIPredicate::slt;
          break;
        case CmpIPredicate::ult:
          // (float)int < -4.4   --> false
          // (float)int < 4.4    --> int <= 4
          if (rhs.isNegative()) {
            rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/false,
                                                       /*width=*/1);
            return success();
          }
          pred = CmpIPredicate::ule;
          break;
        case CmpIPredicate::slt:
          // (float)int < -4.4   --> int < -4
          // (float)int < 4.4    --> int <= 4
          if (!rhs.isNegative())
            pred = CmpIPredicate::sle;
          break;
        case CmpIPredicate::ugt:
          // (float)int > 4.4    --> int > 4
          // (float)int > -4.4   --> true
          if (rhs.isNegative()) {
            rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/true,
                                                       /*width=*/1);
            return success();
          }
          break;
        case CmpIPredicate::sgt:
          // (float)int > 4.4    --> int > 4
          // (float)int > -4.4   --> int >= -4
          if (rhs.isNegative())
            pred = CmpIPredicate::sge;
          break;
        case CmpIPredicate::uge:
          // (float)int >= -4.4   --> true
          // (float)int >= 4.4    --> int > 4
          if (rhs.isNegative()) {
            rewriter.replaceOpWithNewOp<ConstantIntOp>(op, /*value=*/true,
                                                       /*width=*/1);
            return success();
          }
          pred = CmpIPredicate::ugt;
          break;
        case CmpIPredicate::sge:
          // (float)int >= -4.4   --> int >= -4
          // (float)int >= 4.4    --> int > 4
          if (!rhs.isNegative())
            pred = CmpIPredicate::sgt;
          break;
        }
      }
    }

    // Lower this FP comparison into an appropriate integer version of the
    // comparison.
    rewriter.replaceOpWithNewOp<CmpIOp>(
        op, pred, intVal,
        rewriter.create<ConstantOp>(
            op.getLoc(), intVal.getType(),
            rewriter.getIntegerAttr(intVal.getType(), rhsInt)));
    return success();
  }
};

void arith::CmpFOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.insert<CmpFIntToFPConst>(context);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// Transforms a select of a boolean to arithmetic operations
//
//  arith.select %arg, %x, %y : i1
//
//  becomes
//
//  and(%arg, %x) or and(!%arg, %y)
struct SelectI1Simplify : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isInteger(1))
      return failure();

    Value falseConstant =
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), true, 1);
    Value notCondition = rewriter.create<arith::XOrIOp>(
        op.getLoc(), op.getCondition(), falseConstant);

    Value trueVal = rewriter.create<arith::AndIOp>(
        op.getLoc(), op.getCondition(), op.getTrueValue());
    Value falseVal = rewriter.create<arith::AndIOp>(op.getLoc(), notCondition,
                                                    op.getFalseValue());
    rewriter.replaceOpWithNewOp<arith::OrIOp>(op, trueVal, falseVal);
    return success();
  }
};

//  select %arg, %c1, %c0 => extui %arg
struct SelectToExtUI : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    // Cannot extui i1 to i1, or i1 to f32
    if (!op.getType().isa<IntegerType>() || op.getType().isInteger(1))
      return failure();

    // select %x, c1, %c0 => extui %arg
    if (matchPattern(op.getTrueValue(), m_One()) &&
        matchPattern(op.getFalseValue(), m_Zero())) {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, op.getType(),
                                                  op.getCondition());
      return success();
    }

    // select %x, c0, %c1 => extui (xor %arg, true)
    if (matchPattern(op.getTrueValue(), m_Zero()) &&
        matchPattern(op.getFalseValue(), m_One())) {
      rewriter.replaceOpWithNewOp<arith::ExtUIOp>(
          op, op.getType(),
          rewriter.create<arith::XOrIOp>(
              op.getLoc(), op.getCondition(),
              rewriter.create<arith::ConstantIntOp>(
                  op.getLoc(), 1, op.getCondition().getType())));
      return success();
    }

    return failure();
  }
};

void arith::SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<SelectI1Simplify, SelectToExtUI>(context);
}

OpFoldResult arith::SelectOp::fold(ArrayRef<Attribute> operands) {
  Value trueVal = getTrueValue();
  Value falseVal = getFalseValue();
  if (trueVal == falseVal)
    return trueVal;

  Value condition = getCondition();

  // select true, %0, %1 => %0
  if (matchPattern(condition, m_One()))
    return trueVal;

  // select false, %0, %1 => %1
  if (matchPattern(condition, m_Zero()))
    return falseVal;

  // select %x, true, false => %x
  if (getType().isInteger(1) && matchPattern(getTrueValue(), m_One()) &&
      matchPattern(getFalseValue(), m_Zero()))
    return condition;

  if (auto cmp = dyn_cast_or_null<arith::CmpIOp>(condition.getDefiningOp())) {
    auto pred = cmp.getPredicate();
    if (pred == arith::CmpIPredicate::eq || pred == arith::CmpIPredicate::ne) {
      auto cmpLhs = cmp.getLhs();
      auto cmpRhs = cmp.getRhs();

      // %0 = arith.cmpi eq, %arg0, %arg1
      // %1 = arith.select %0, %arg0, %arg1 => %arg1

      // %0 = arith.cmpi ne, %arg0, %arg1
      // %1 = arith.select %0, %arg0, %arg1 => %arg0

      if ((cmpLhs == trueVal && cmpRhs == falseVal) ||
          (cmpRhs == trueVal && cmpLhs == falseVal))
        return pred == arith::CmpIPredicate::ne ? trueVal : falseVal;
    }
  }
  return nullptr;
}

ParseResult SelectOp::parse(OpAsmParser &parser, OperationState &result) {
  Type conditionType, resultType;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType))
    return failure();

  // Check for the explicit condition type if this is a masked tensor or vector.
  if (succeeded(parser.parseOptionalComma())) {
    conditionType = resultType;
    if (parser.parseType(resultType))
      return failure();
  } else {
    conditionType = parser.getBuilder().getI1Type();
  }

  result.addTypes(resultType);
  return parser.resolveOperands(operands,
                                {conditionType, resultType, resultType},
                                parser.getNameLoc(), result.operands);
}

void arith::SelectOp::print(OpAsmPrinter &p) {
  p << " " << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : ";
  if (ShapedType condType = getCondition().getType().dyn_cast<ShapedType>())
    p << condType << ", ";
  p << getType();
}

LogicalResult arith::SelectOp::verify() {
  Type conditionType = getCondition().getType();
  if (conditionType.isSignlessInteger(1))
    return success();

  // If the result type is a vector or tensor, the type can be a mask with the
  // same elements.
  Type resultType = getType();
  if (!resultType.isa<TensorType, VectorType>())
    return emitOpError() << "expected condition to be a signless i1, but got "
                         << conditionType;
  Type shapedConditionType = getI1SameShape(resultType);
  if (conditionType != shapedConditionType) {
    return emitOpError() << "expected condition type to have the same shape "
                            "as the result type, expected "
                         << shapedConditionType << ", but got "
                         << conditionType;
  }
  return success();
}
//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ShLIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if shifting more than the bit width.
  bool bounded = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) {
        bounded = b.ule(b.getBitWidth());
        return a.shl(b);
      });
  return bounded ? result : Attribute();
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ShRUIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if shifting more than the bit width.
  bool bounded = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) {
        bounded = b.ule(b.getBitWidth());
        return a.lshr(b);
      });
  return bounded ? result : Attribute();
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ShRSIOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if shifting more than the bit width.
  bool bounded = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      operands, [&](const APInt &a, const APInt &b) {
        bounded = b.ule(b.getBitWidth());
        return a.ashr(b);
      });
  return bounded ? result : Attribute();
}

//===----------------------------------------------------------------------===//
// Atomic Enum
//===----------------------------------------------------------------------===//

/// Returns the identity value attribute associated with an AtomicRMWKind op.
Attribute mlir::arith::getIdentityValueAttr(AtomicRMWKind kind, Type resultType,
                                            OpBuilder &builder, Location loc) {
  switch (kind) {
  case AtomicRMWKind::maxf:
    return builder.getFloatAttr(
        resultType,
        APFloat::getInf(resultType.cast<FloatType>().getFloatSemantics(),
                        /*Negative=*/true));
  case AtomicRMWKind::addf:
  case AtomicRMWKind::addi:
  case AtomicRMWKind::maxu:
  case AtomicRMWKind::ori:
    return builder.getZeroAttr(resultType);
  case AtomicRMWKind::andi:
    return builder.getIntegerAttr(
        resultType,
        APInt::getAllOnes(resultType.cast<IntegerType>().getWidth()));
  case AtomicRMWKind::maxs:
    return builder.getIntegerAttr(
        resultType,
        APInt::getSignedMinValue(resultType.cast<IntegerType>().getWidth()));
  case AtomicRMWKind::minf:
    return builder.getFloatAttr(
        resultType,
        APFloat::getInf(resultType.cast<FloatType>().getFloatSemantics(),
                        /*Negative=*/false));
  case AtomicRMWKind::mins:
    return builder.getIntegerAttr(
        resultType,
        APInt::getSignedMaxValue(resultType.cast<IntegerType>().getWidth()));
  case AtomicRMWKind::minu:
    return builder.getIntegerAttr(
        resultType,
        APInt::getMaxValue(resultType.cast<IntegerType>().getWidth()));
  case AtomicRMWKind::muli:
    return builder.getIntegerAttr(resultType, 1);
  case AtomicRMWKind::mulf:
    return builder.getFloatAttr(resultType, 1);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
}

/// Returns the identity value associated with an AtomicRMWKind op.
Value mlir::arith::getIdentityValue(AtomicRMWKind op, Type resultType,
                                    OpBuilder &builder, Location loc) {
  Attribute attr = getIdentityValueAttr(op, resultType, builder, loc);
  return builder.create<arith::ConstantOp>(loc, attr);
}

/// Return the value obtained by applying the reduction operation kind
/// associated with a binary AtomicRMWKind op to `lhs` and `rhs`.
Value mlir::arith::getReductionOp(AtomicRMWKind op, OpBuilder &builder,
                                  Location loc, Value lhs, Value rhs) {
  switch (op) {
  case AtomicRMWKind::addf:
    return builder.create<arith::AddFOp>(loc, lhs, rhs);
  case AtomicRMWKind::addi:
    return builder.create<arith::AddIOp>(loc, lhs, rhs);
  case AtomicRMWKind::mulf:
    return builder.create<arith::MulFOp>(loc, lhs, rhs);
  case AtomicRMWKind::muli:
    return builder.create<arith::MulIOp>(loc, lhs, rhs);
  case AtomicRMWKind::maxf:
    return builder.create<arith::MaxFOp>(loc, lhs, rhs);
  case AtomicRMWKind::minf:
    return builder.create<arith::MinFOp>(loc, lhs, rhs);
  case AtomicRMWKind::maxs:
    return builder.create<arith::MaxSIOp>(loc, lhs, rhs);
  case AtomicRMWKind::mins:
    return builder.create<arith::MinSIOp>(loc, lhs, rhs);
  case AtomicRMWKind::maxu:
    return builder.create<arith::MaxUIOp>(loc, lhs, rhs);
  case AtomicRMWKind::minu:
    return builder.create<arith::MinUIOp>(loc, lhs, rhs);
  case AtomicRMWKind::ori:
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  case AtomicRMWKind::andi:
    return builder.create<arith::AndIOp>(loc, lhs, rhs);
  // TODO: Add remaining reduction operations.
  default:
    (void)emitOptionalError(loc, "Reduction operation type not supported");
    break;
  }
  return nullptr;
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
