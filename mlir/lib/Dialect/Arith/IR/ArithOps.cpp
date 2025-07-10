//===- ArithOps.cpp - MLIR Arith dialect ops implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>
#include <functional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Pattern helpers
//===----------------------------------------------------------------------===//

static IntegerAttr
applyToIntegerAttrs(PatternRewriter &builder, Value res, Attribute lhs,
                    Attribute rhs,
                    function_ref<APInt(const APInt &, const APInt &)> binFn) {
  APInt lhsVal = llvm::cast<IntegerAttr>(lhs).getValue();
  APInt rhsVal = llvm::cast<IntegerAttr>(rhs).getValue();
  APInt value = binFn(lhsVal, rhsVal);
  return IntegerAttr::get(res.getType(), value);
}

static IntegerAttr addIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::plus<APInt>());
}

static IntegerAttr subIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::minus<APInt>());
}

static IntegerAttr mulIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::multiplies<APInt>());
}

// Merge overflow flags from 2 ops, selecting the most conservative combination.
static IntegerOverflowFlagsAttr
mergeOverflowFlags(IntegerOverflowFlagsAttr val1,
                   IntegerOverflowFlagsAttr val2) {
  return IntegerOverflowFlagsAttr::get(val1.getContext(),
                                       val1.getValue() & val2.getValue());
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

/// Equivalent to
/// convertRoundingModeToLLVM(convertArithRoundingModeToLLVM(roundingMode)).
///
/// Not possible to implement as chain of calls as this would introduce a
/// circular dependency with MLIRArithAttrToLLVMConversion and make arith depend
/// on the LLVM dialect and on translation to LLVM.
static llvm::RoundingMode
convertArithRoundingModeToLLVMIR(RoundingMode roundingMode) {
  switch (roundingMode) {
  case RoundingMode::downward:
    return llvm::RoundingMode::TowardNegative;
  case RoundingMode::to_nearest_away:
    return llvm::RoundingMode::NearestTiesToAway;
  case RoundingMode::to_nearest_even:
    return llvm::RoundingMode::NearestTiesToEven;
  case RoundingMode::toward_zero:
    return llvm::RoundingMode::TowardZero;
  case RoundingMode::upward:
    return llvm::RoundingMode::TowardPositive;
  }
  llvm_unreachable("Unhandled rounding mode");
}

static arith::CmpIPredicateAttr invertPredicate(arith::CmpIPredicateAttr pred) {
  return arith::CmpIPredicateAttr::get(pred.getContext(),
                                       invertPredicate(pred.getValue()));
}

static int64_t getScalarOrElementWidth(Type type) {
  Type elemTy = getElementTypeOrSelf(type);
  if (elemTy.isIntOrFloat())
    return elemTy.getIntOrFloatBitWidth();

  return -1;
}

static int64_t getScalarOrElementWidth(Value value) {
  return getScalarOrElementWidth(value.getType());
}

static FailureOr<APInt> getIntOrSplatIntValue(Attribute attr) {
  APInt value;
  if (matchPattern(attr, m_ConstantInt(&value)))
    return value;

  return failure();
}

static Attribute getBoolAttribute(Type type, bool value) {
  auto boolAttr = BoolAttr::get(type.getContext(), value);
  ShapedType shapedType = llvm::dyn_cast_or_null<ShapedType>(type);
  if (!shapedType)
    return boolAttr;
  return DenseElementsAttr::get(shapedType, boolAttr);
}

//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "ArithCanonicalization.inc"
} // namespace

//===----------------------------------------------------------------------===//
// Common helpers
//===----------------------------------------------------------------------===//

/// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto shapedType = llvm::dyn_cast<ShapedType>(type))
    return shapedType.cloneWith(std::nullopt, i1Type);
  if (llvm::isa<UnrankedTensorType>(type))
    return UnrankedTensorType::get(i1Type);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void arith::ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto type = getType();
  if (auto intCst = llvm::dyn_cast<IntegerAttr>(getValue())) {
    auto intType = llvm::dyn_cast<IntegerType>(type);

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
  // Integer values must be signless.
  if (llvm::isa<IntegerType>(type) &&
      !llvm::cast<IntegerType>(type).isSignless())
    return emitOpError("integer return type must be signless");
  // Any float or elements attribute are acceptable.
  if (!llvm::isa<IntegerAttr, FloatAttr, ElementsAttr>(getValue())) {
    return emitOpError(
        "value must be an integer, float, or elements attribute");
  }

  // Note, we could relax this for vectors with 1 scalable dim, e.g.:
  //  * arith.constant dense<[[3, 3], [1, 1]]> : vector<2 x [2] x i32>
  // However, this would most likely require updating the lowerings to LLVM.
  if (isa<ScalableVectorType>(type) && !isa<SplatElementsAttr>(getValue()))
    return emitOpError(
        "intializing scalable vectors with elements attribute is not supported"
        " unless it's a vector splat");
  return success();
}

bool arith::ConstantOp::isBuildableWith(Attribute value, Type type) {
  // The value's type must be the same as the provided type.
  auto typedAttr = llvm::dyn_cast<TypedAttr>(value);
  if (!typedAttr || typedAttr.getType() != type)
    return false;
  // Integer values must be signless.
  if (llvm::isa<IntegerType>(type) &&
      !llvm::cast<IntegerType>(type).isSignless())
    return false;
  // Integer, float, and element attributes are buildable.
  return llvm::isa<IntegerAttr, FloatAttr, ElementsAttr>(value);
}

ConstantOp arith::ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  if (isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, cast<TypedAttr>(value));
  return nullptr;
}

OpFoldResult arith::ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

void arith::ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                                 int64_t value, unsigned width) {
  auto type = builder.getIntegerType(width);
  arith::ConstantOp::build(builder, result, type,
                           builder.getIntegerAttr(type, value));
}

void arith::ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                                 Type type, int64_t value) {
  arith::ConstantOp::build(builder, result, type,
                           builder.getIntegerAttr(type, value));
}

void arith::ConstantIntOp::build(OpBuilder &builder, OperationState &result,
                                 Type type, const APInt &value) {
  arith::ConstantOp::build(builder, result, type,
                           builder.getIntegerAttr(type, value));
}

bool arith::ConstantIntOp::classof(Operation *op) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op))
    return constOp.getType().isSignlessInteger();
  return false;
}

void arith::ConstantFloatOp::build(OpBuilder &builder, OperationState &result,
                                   FloatType type, const APFloat &value) {
  arith::ConstantOp::build(builder, result, type,
                           builder.getFloatAttr(type, value));
}

bool arith::ConstantFloatOp::classof(Operation *op) {
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(op))
    return llvm::isa<FloatType>(constOp.getType());
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

Value mlir::arith::getZeroConstant(OpBuilder &builder, Location loc,
                                   Type type) {
  // TODO: Incorporate this check to `FloatAttr::get*`.
  assert(!isa<Float8E8M0FNUType>(getElementTypeOrSelf(type)) &&
         "type doesn't have a zero representation");
  TypedAttr zeroAttr = builder.getZeroAttr(type);
  assert(zeroAttr && "unsupported type for zero attribute");
  return builder.create<arith::ConstantOp>(loc, zeroAttr);
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AddIOp::fold(FoldAdaptor adaptor) {
  // addi(x, 0) -> x
  if (matchPattern(adaptor.getRhs(), m_Zero()))
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
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) + b; });
}

void arith::AddIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<AddIAddConstant, AddISubConstantRHS, AddISubConstantLHS,
               AddIMulNegativeOneRhs, AddIMulNegativeOneLhs>(context);
}

//===----------------------------------------------------------------------===//
// AddUIExtendedOp
//===----------------------------------------------------------------------===//

std::optional<SmallVector<int64_t, 4>>
arith::AddUIExtendedOp::getShapeForUnroll() {
  if (auto vt = llvm::dyn_cast<VectorType>(getType(0)))
    return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

// Returns the overflow bit, assuming that `sum` is the result of unsigned
// addition of `operand` and another number.
static APInt calculateUnsignedOverflow(const APInt &sum, const APInt &operand) {
  return sum.ult(operand) ? APInt::getAllOnes(1) : APInt::getZero(1);
}

LogicalResult
arith::AddUIExtendedOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<OpFoldResult> &results) {
  Type overflowTy = getOverflow().getType();
  // addui_extended(x, 0) -> x, false
  if (matchPattern(getRhs(), m_Zero())) {
    Builder builder(getContext());
    auto falseValue = builder.getZeroAttr(overflowTy);

    results.push_back(getLhs());
    results.push_back(falseValue);
    return success();
  }

  // addui_extended(constant_a, constant_b) -> constant_sum, constant_carry
  // Let the `constFoldBinaryOp` utility attempt to fold the sum of both
  // operands. If that succeeds, calculate the overflow bit based on the sum
  // and the first (constant) operand, `lhs`.
  if (Attribute sumAttr = constFoldBinaryOp<IntegerAttr>(
          adaptor.getOperands(),
          [](APInt a, const APInt &b) { return std::move(a) + b; })) {
    Attribute overflowAttr = constFoldBinaryOp<IntegerAttr>(
        ArrayRef({sumAttr, adaptor.getLhs()}),
        getI1SameShape(llvm::cast<TypedAttr>(sumAttr).getType()),
        calculateUnsignedOverflow);
    if (!overflowAttr)
      return failure();

    results.push_back(sumAttr);
    results.push_back(overflowAttr);
    return success();
  }

  return failure();
}

void arith::AddUIExtendedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<AddUIExtendedToAddI>(context);
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::SubIOp::fold(FoldAdaptor adaptor) {
  // subi(x,x) -> 0
  if (getOperand(0) == getOperand(1)) {
    auto shapedType = dyn_cast<ShapedType>(getType());
    // We can't generate a constant with a dynamic shaped tensor.
    if (!shapedType || shapedType.hasStaticShape())
      return Builder(getContext()).getZeroAttr(getType());
  }
  // subi(x,0) -> x
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();

  if (auto add = getLhs().getDefiningOp<AddIOp>()) {
    // subi(addi(a, b), b) -> a
    if (getRhs() == add.getRhs())
      return add.getLhs();
    // subi(addi(a, b), a) -> b
    if (getRhs() == add.getLhs())
      return add.getRhs();
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) - b; });
}

void arith::SubIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<SubIRHSAddConstant, SubILHSAddConstant, SubIRHSSubConstantRHS,
               SubIRHSSubConstantLHS, SubILHSSubConstantRHS,
               SubILHSSubConstantLHS, SubISubILHSRHSLHS>(context);
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MulIOp::fold(FoldAdaptor adaptor) {
  // muli(x, 0) -> 0
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getRhs();
  // muli(x, 1) -> x
  if (matchPattern(adaptor.getRhs(), m_One()))
    return getLhs();
  // TODO: Handle the overflow case.

  // default folder
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a, const APInt &b) { return a * b; });
}

void arith::MulIOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  if (!isa<IndexType>(getType()))
    return;

  // Match vector.vscale by name to avoid depending on the vector dialect (which
  // is a circular dependency).
  auto isVscale = [](Operation *op) {
    return op && op->getName().getStringRef() == "vector.vscale";
  };

  IntegerAttr baseValue;
  auto isVscaleExpr = [&](Value a, Value b) {
    return matchPattern(a, m_Constant(&baseValue)) &&
           isVscale(b.getDefiningOp());
  };

  if (!isVscaleExpr(getLhs(), getRhs()) && !isVscaleExpr(getRhs(), getLhs()))
    return;

  // Name `base * vscale` or `vscale * base` as `c<base_value>_vscale`.
  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << baseValue.getInt() << "_vscale";
  setNameFn(getResult(), specialName.str());
}

void arith::MulIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<MulIMulIConstant>(context);
}

//===----------------------------------------------------------------------===//
// MulSIExtendedOp
//===----------------------------------------------------------------------===//

std::optional<SmallVector<int64_t, 4>>
arith::MulSIExtendedOp::getShapeForUnroll() {
  if (auto vt = llvm::dyn_cast<VectorType>(getType(0)))
    return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

LogicalResult
arith::MulSIExtendedOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<OpFoldResult> &results) {
  // mulsi_extended(x, 0) -> 0, 0
  if (matchPattern(adaptor.getRhs(), m_Zero())) {
    Attribute zero = adaptor.getRhs();
    results.push_back(zero);
    results.push_back(zero);
    return success();
  }

  // mulsi_extended(cst_a, cst_b) -> cst_low, cst_high
  if (Attribute lowAttr = constFoldBinaryOp<IntegerAttr>(
          adaptor.getOperands(),
          [](const APInt &a, const APInt &b) { return a * b; })) {
    // Invoke the constant fold helper again to calculate the 'high' result.
    Attribute highAttr = constFoldBinaryOp<IntegerAttr>(
        adaptor.getOperands(), [](const APInt &a, const APInt &b) {
          return llvm::APIntOps::mulhs(a, b);
        });
    assert(highAttr && "Unexpected constant-folding failure");

    results.push_back(lowAttr);
    results.push_back(highAttr);
    return success();
  }

  return failure();
}

void arith::MulSIExtendedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<MulSIExtendedToMulI, MulSIExtendedRHSOne>(context);
}

//===----------------------------------------------------------------------===//
// MulUIExtendedOp
//===----------------------------------------------------------------------===//

std::optional<SmallVector<int64_t, 4>>
arith::MulUIExtendedOp::getShapeForUnroll() {
  if (auto vt = llvm::dyn_cast<VectorType>(getType(0)))
    return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

LogicalResult
arith::MulUIExtendedOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<OpFoldResult> &results) {
  // mului_extended(x, 0) -> 0, 0
  if (matchPattern(adaptor.getRhs(), m_Zero())) {
    Attribute zero = adaptor.getRhs();
    results.push_back(zero);
    results.push_back(zero);
    return success();
  }

  // mului_extended(x, 1) -> x, 0
  if (matchPattern(adaptor.getRhs(), m_One())) {
    Builder builder(getContext());
    Attribute zero = builder.getZeroAttr(getLhs().getType());
    results.push_back(getLhs());
    results.push_back(zero);
    return success();
  }

  // mului_extended(cst_a, cst_b) -> cst_low, cst_high
  if (Attribute lowAttr = constFoldBinaryOp<IntegerAttr>(
          adaptor.getOperands(),
          [](const APInt &a, const APInt &b) { return a * b; })) {
    // Invoke the constant fold helper again to calculate the 'high' result.
    Attribute highAttr = constFoldBinaryOp<IntegerAttr>(
        adaptor.getOperands(), [](const APInt &a, const APInt &b) {
          return llvm::APIntOps::mulhu(a, b);
        });
    assert(highAttr && "Unexpected constant-folding failure");

    results.push_back(lowAttr);
    results.push_back(highAttr);
    return success();
  }

  return failure();
}

void arith::MulUIExtendedOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<MulUIExtendedToMulI>(context);
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

/// Fold `(a * b) / b -> a`
static Value foldDivMul(Value lhs, Value rhs,
                        arith::IntegerOverflowFlags ovfFlags) {
  auto mul = lhs.getDefiningOp<mlir::arith::MulIOp>();
  if (!mul || !bitEnumContainsAll(mul.getOverflowFlags(), ovfFlags))
    return {};

  if (mul.getLhs() == rhs)
    return mul.getRhs();

  if (mul.getRhs() == rhs)
    return mul.getLhs();

  return {};
}

OpFoldResult arith::DivUIOp::fold(FoldAdaptor adaptor) {
  // divui (x, 1) -> x.
  if (matchPattern(adaptor.getRhs(), m_One()))
    return getLhs();

  // (a * b) / b -> a
  if (Value val = foldDivMul(getLhs(), getRhs(), IntegerOverflowFlags::nuw))
    return val;

  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(),
                                               [&](APInt a, const APInt &b) {
                                                 if (div0 || !b) {
                                                   div0 = true;
                                                   return a;
                                                 }
                                                 return a.udiv(b);
                                               });

  return div0 ? Attribute() : result;
}

/// Returns whether an unsigned division by `divisor` is speculatable.
static Speculation::Speculatability getDivUISpeculatability(Value divisor) {
  // X / 0 => UB
  if (matchPattern(divisor, m_IntRangeWithoutZeroU()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

Speculation::Speculatability arith::DivUIOp::getSpeculatability() {
  return getDivUISpeculatability(getRhs());
}

//===----------------------------------------------------------------------===//
// DivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivSIOp::fold(FoldAdaptor adaptor) {
  // divsi (x, 1) -> x.
  if (matchPattern(adaptor.getRhs(), m_One()))
    return getLhs();

  // (a * b) / b -> a
  if (Value val = foldDivMul(getLhs(), getRhs(), IntegerOverflowFlags::nsw))
    return val;

  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](APInt a, const APInt &b) {
        if (overflowOrDiv0 || !b) {
          overflowOrDiv0 = true;
          return a;
        }
        return a.sdiv_ov(b, overflowOrDiv0);
      });

  return overflowOrDiv0 ? Attribute() : result;
}

/// Returns whether a signed division by `divisor` is speculatable. This
/// function conservatively assumes that all signed division by -1 are not
/// speculatable.
static Speculation::Speculatability getDivSISpeculatability(Value divisor) {
  // X / 0 => UB
  // INT_MIN / -1 => UB
  if (matchPattern(divisor, m_IntRangeWithoutZeroS()) &&
      matchPattern(divisor, m_IntRangeWithoutNegOneS()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

Speculation::Speculatability arith::DivSIOp::getSpeculatability() {
  return getDivSISpeculatability(getRhs());
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

OpFoldResult arith::CeilDivUIOp::fold(FoldAdaptor adaptor) {
  // ceildivui (x, 1) -> x.
  if (matchPattern(adaptor.getRhs(), m_One()))
    return getLhs();

  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](APInt a, const APInt &b) {
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

Speculation::Speculatability arith::CeilDivUIOp::getSpeculatability() {
  return getDivUISpeculatability(getRhs());
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::CeilDivSIOp::fold(FoldAdaptor adaptor) {
  // ceildivsi (x, 1) -> x.
  if (matchPattern(adaptor.getRhs(), m_One()))
    return getLhs();

  // Don't fold if it would overflow or if it requires a division by zero.
  // TODO: This hook won't fold operations where a = MININT, because
  // negating MININT overflows. This can be improved.
  bool overflowOrDiv0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](APInt a, const APInt &b) {
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

        // No folding happens if any of the intermediate arithmetic operations
        // overflows.
        bool overflowNegA = false;
        bool overflowNegB = false;
        bool overflowDiv = false;
        bool overflowNegRes = false;
        if (!aGtZero && !bGtZero) {
          // Both negative, return ceil(-a, -b).
          APInt posA = zero.ssub_ov(a, overflowNegA);
          APInt posB = zero.ssub_ov(b, overflowNegB);
          APInt res = signedCeilNonnegInputs(posA, posB, overflowDiv);
          overflowOrDiv0 = (overflowNegA || overflowNegB || overflowDiv);
          return res;
        }
        if (!aGtZero && bGtZero) {
          // A is negative, b is positive, return - ( -a / b).
          APInt posA = zero.ssub_ov(a, overflowNegA);
          APInt div = posA.sdiv_ov(b, overflowDiv);
          APInt res = zero.ssub_ov(div, overflowNegRes);
          overflowOrDiv0 = (overflowNegA || overflowDiv || overflowNegRes);
          return res;
        }
        // A is positive, b is negative, return - (a / -b).
        APInt posB = zero.ssub_ov(b, overflowNegB);
        APInt div = a.sdiv_ov(posB, overflowDiv);
        APInt res = zero.ssub_ov(div, overflowNegRes);

        overflowOrDiv0 = (overflowNegB || overflowDiv || overflowNegRes);
        return res;
      });

  return overflowOrDiv0 ? Attribute() : result;
}

Speculation::Speculatability arith::CeilDivSIOp::getSpeculatability() {
  return getDivSISpeculatability(getRhs());
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::FloorDivSIOp::fold(FoldAdaptor adaptor) {
  // floordivsi (x, 1) -> x.
  if (matchPattern(adaptor.getRhs(), m_One()))
    return getLhs();

  // Don't fold if it would overflow or if it requires a division by zero.
  bool overflowOrDiv = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](APInt a, const APInt &b) {
        if (b.isZero()) {
          overflowOrDiv = true;
          return a;
        }
        return a.sfloordiv_ov(b, overflowOrDiv);
      });

  return overflowOrDiv ? Attribute() : result;
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemUIOp::fold(FoldAdaptor adaptor) {
  // remui (x, 1) -> 0.
  if (matchPattern(adaptor.getRhs(), m_One()))
    return Builder(getContext()).getZeroAttr(getType());

  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(),
                                               [&](APInt a, const APInt &b) {
                                                 if (div0 || b.isZero()) {
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

OpFoldResult arith::RemSIOp::fold(FoldAdaptor adaptor) {
  // remsi (x, 1) -> 0.
  if (matchPattern(adaptor.getRhs(), m_One()))
    return Builder(getContext()).getZeroAttr(getType());

  // Don't fold if it would require a division by zero.
  bool div0 = false;
  auto result = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(),
                                               [&](APInt a, const APInt &b) {
                                                 if (div0 || b.isZero()) {
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

/// Fold `and(a, and(a, b))` to `and(a, b)`
static Value foldAndIofAndI(arith::AndIOp op) {
  for (bool reversePrev : {false, true}) {
    auto prev = (reversePrev ? op.getRhs() : op.getLhs())
                    .getDefiningOp<arith::AndIOp>();
    if (!prev)
      continue;

    Value other = (reversePrev ? op.getLhs() : op.getRhs());
    if (other != prev.getLhs() && other != prev.getRhs())
      continue;

    return prev.getResult();
  }
  return {};
}

OpFoldResult arith::AndIOp::fold(FoldAdaptor adaptor) {
  /// and(x, 0) -> 0
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getRhs();
  /// and(x, allOnes) -> x
  APInt intValue;
  if (matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue)) &&
      intValue.isAllOnes())
    return getLhs();
  /// and(x, not(x)) -> 0
  if (matchPattern(getRhs(), m_Op<XOrIOp>(matchers::m_Val(getLhs()),
                                          m_ConstantInt(&intValue))) &&
      intValue.isAllOnes())
    return Builder(getContext()).getZeroAttr(getType());
  /// and(not(x), x) -> 0
  if (matchPattern(getLhs(), m_Op<XOrIOp>(matchers::m_Val(getRhs()),
                                          m_ConstantInt(&intValue))) &&
      intValue.isAllOnes())
    return Builder(getContext()).getZeroAttr(getType());

  /// and(a, and(a, b)) -> and(a, b)
  if (Value result = foldAndIofAndI(*this))
    return result;

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) & b; });
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::OrIOp::fold(FoldAdaptor adaptor) {
  if (APInt rhsVal; matchPattern(adaptor.getRhs(), m_ConstantInt(&rhsVal))) {
    /// or(x, 0) -> x
    if (rhsVal.isZero())
      return getLhs();
    /// or(x, <all ones>) -> <all ones>
    if (rhsVal.isAllOnes())
      return adaptor.getRhs();
  }

  APInt intValue;
  /// or(x, xor(x, 1)) -> 1
  if (matchPattern(getRhs(), m_Op<XOrIOp>(matchers::m_Val(getLhs()),
                                          m_ConstantInt(&intValue))) &&
      intValue.isAllOnes())
    return getRhs().getDefiningOp<XOrIOp>().getRhs();
  /// or(xor(x, 1), x) -> 1
  if (matchPattern(getLhs(), m_Op<XOrIOp>(matchers::m_Val(getRhs()),
                                          m_ConstantInt(&intValue))) &&
      intValue.isAllOnes())
    return getLhs().getDefiningOp<XOrIOp>().getRhs();

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) | b; });
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::XOrIOp::fold(FoldAdaptor adaptor) {
  /// xor(x, 0) -> x
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();
  /// xor(x, x) -> 0
  if (getLhs() == getRhs())
    return Builder(getContext()).getZeroAttr(getType());
  /// xor(xor(x, a), a) -> x
  /// xor(xor(a, x), a) -> x
  if (arith::XOrIOp prev = getLhs().getDefiningOp<arith::XOrIOp>()) {
    if (prev.getRhs() == getRhs())
      return prev.getLhs();
    if (prev.getLhs() == getRhs())
      return prev.getRhs();
  }
  /// xor(a, xor(x, a)) -> x
  /// xor(a, xor(a, x)) -> x
  if (arith::XOrIOp prev = getRhs().getDefiningOp<arith::XOrIOp>()) {
    if (prev.getRhs() == getLhs())
      return prev.getLhs();
    if (prev.getLhs() == getLhs())
      return prev.getRhs();
  }

  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) ^ b; });
}

void arith::XOrIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<XOrINotCmpI, XOrIOfExtUI, XOrIOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::NegFOp::fold(FoldAdaptor adaptor) {
  /// negf(negf(x)) -> x
  if (auto op = this->getOperand().getDefiningOp<arith::NegFOp>())
    return op.getOperand();
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(),
                                     [](const APFloat &a) { return -a; });
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::AddFOp::fold(FoldAdaptor adaptor) {
  // addf(x, -0) -> x
  if (matchPattern(adaptor.getRhs(), m_NegZeroFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) { return a + b; });
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::SubFOp::fold(FoldAdaptor adaptor) {
  // subf(x, +0) -> x
  if (matchPattern(adaptor.getRhs(), m_PosZeroFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) { return a - b; });
}

//===----------------------------------------------------------------------===//
// MaximumFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MaximumFOp::fold(FoldAdaptor adaptor) {
  // maximumf(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  // maximumf(x, -inf) -> x
  if (matchPattern(adaptor.getRhs(), m_NegInfFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) { return llvm::maximum(a, b); });
}

//===----------------------------------------------------------------------===//
// MaxNumFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MaxNumFOp::fold(FoldAdaptor adaptor) {
  // maxnumf(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  // maxnumf(x, NaN) -> x
  if (matchPattern(adaptor.getRhs(), m_NaNFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), llvm::maxnum);
}

//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

OpFoldResult MaxSIOp::fold(FoldAdaptor adaptor) {
  // maxsi(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  if (APInt intValue;
      matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue))) {
    // maxsi(x,MAX_INT) -> MAX_INT
    if (intValue.isMaxSignedValue())
      return getRhs();
    // maxsi(x, MIN_INT) -> x
    if (intValue.isMinSignedValue())
      return getLhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(),
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::smax(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

OpFoldResult MaxUIOp::fold(FoldAdaptor adaptor) {
  // maxui(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  if (APInt intValue;
      matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue))) {
    // maxui(x,MAX_INT) -> MAX_INT
    if (intValue.isMaxValue())
      return getRhs();
    // maxui(x, MIN_INT) -> x
    if (intValue.isMinValue())
      return getLhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(),
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::umax(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MinimumFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MinimumFOp::fold(FoldAdaptor adaptor) {
  // minimumf(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  // minimumf(x, +inf) -> x
  if (matchPattern(adaptor.getRhs(), m_PosInfFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) { return llvm::minimum(a, b); });
}

//===----------------------------------------------------------------------===//
// MinNumFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MinNumFOp::fold(FoldAdaptor adaptor) {
  // minnumf(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  // minnumf(x, NaN) -> x
  if (matchPattern(adaptor.getRhs(), m_NaNFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) { return llvm::minnum(a, b); });
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

OpFoldResult MinSIOp::fold(FoldAdaptor adaptor) {
  // minsi(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  if (APInt intValue;
      matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue))) {
    // minsi(x,MIN_INT) -> MIN_INT
    if (intValue.isMinSignedValue())
      return getRhs();
    // minsi(x, MAX_INT) -> x
    if (intValue.isMaxSignedValue())
      return getLhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(),
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::smin(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

OpFoldResult MinUIOp::fold(FoldAdaptor adaptor) {
  // minui(x,x) -> x
  if (getLhs() == getRhs())
    return getRhs();

  if (APInt intValue;
      matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue))) {
    // minui(x,MIN_INT) -> MIN_INT
    if (intValue.isMinValue())
      return getRhs();
    // minui(x, MAX_INT) -> x
    if (intValue.isMaxValue())
      return getLhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(),
                                        [](const APInt &a, const APInt &b) {
                                          return llvm::APIntOps::umin(a, b);
                                        });
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::MulFOp::fold(FoldAdaptor adaptor) {
  // mulf(x, 1) -> x
  if (matchPattern(adaptor.getRhs(), m_OneFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) { return a * b; });
}

void arith::MulFOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<MulFOfNegF>(context);
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::DivFOp::fold(FoldAdaptor adaptor) {
  // divf(x, 1) -> x
  if (matchPattern(adaptor.getRhs(), m_OneFloat()))
    return getLhs();

  return constFoldBinaryOp<FloatAttr>(
      adaptor.getOperands(),
      [](const APFloat &a, const APFloat &b) { return a / b; });
}

void arith::DivFOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<DivFOfNegF>(context);
}

//===----------------------------------------------------------------------===//
// RemFOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::RemFOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(),
                                      [](const APFloat &a, const APFloat &b) {
                                        APFloat result(a);
                                        // APFloat::mod() offers the remainder
                                        // behavior we want, i.e. the result has
                                        // the sign of LHS operand.
                                        (void)result.mod(b);
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
  if (llvm::isa<ShapedType>(type) && !llvm::isa<ShapedTypes...>(type))
    return {};

  auto underlyingType = getElementTypeOrSelf(type);
  if (!llvm::isa<ElementTypes...>(underlyingType))
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

/// Return false if both types are ranked tensor with mismatching encoding.
static bool hasSameEncoding(Type typeA, Type typeB) {
  auto rankedTensorA = dyn_cast<RankedTensorType>(typeA);
  auto rankedTensorB = dyn_cast<RankedTensorType>(typeB);
  if (!rankedTensorA || !rankedTensorB)
    return true;
  return rankedTensorA.getEncoding() == rankedTensorB.getEncoding();
}

static bool areValidCastInputsAndOutputs(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  if (!hasSameEncoding(inputs.front(), outputs.front()))
    return false;
  return succeeded(verifyCompatibleShapes(inputs.front(), outputs.front()));
}

//===----------------------------------------------------------------------===//
// Verifiers for integer and floating point extension/truncation ops
//===----------------------------------------------------------------------===//

// Extend ops can only extend to a wider type.
template <typename ValType, typename Op>
static LogicalResult verifyExtOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.getIn().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (llvm::cast<ValType>(srcType).getWidth() >=
      llvm::cast<ValType>(dstType).getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

// Truncate ops can only truncate to a shorter type.
template <typename ValType, typename Op>
static LogicalResult verifyTruncateOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.getIn().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (llvm::cast<ValType>(srcType).getWidth() <=
      llvm::cast<ValType>(dstType).getWidth())
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

/// Attempts to convert `sourceValue` to an APFloat value with
/// `targetSemantics` and `roundingMode`, without any information loss.
static FailureOr<APFloat> convertFloatValue(
    APFloat sourceValue, const llvm::fltSemantics &targetSemantics,
    llvm::RoundingMode roundingMode = llvm::RoundingMode::NearestTiesToEven) {
  bool losesInfo = false;
  auto status = sourceValue.convert(targetSemantics, roundingMode, &losesInfo);
  if (losesInfo || status != APFloat::opOK)
    return failure();

  return sourceValue;
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ExtUIOp::fold(FoldAdaptor adaptor) {
  if (auto lhs = getIn().getDefiningOp<ExtUIOp>()) {
    getInMutable().assign(lhs.getIn());
    return getResult();
  }

  Type resType = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      adaptor.getOperands(), getType(),
      [bitWidth](const APInt &a, bool &castStatus) {
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

OpFoldResult arith::ExtSIOp::fold(FoldAdaptor adaptor) {
  if (auto lhs = getIn().getDefiningOp<ExtSIOp>()) {
    getInMutable().assign(lhs.getIn());
    return getResult();
  }

  Type resType = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      adaptor.getOperands(), getType(),
      [bitWidth](const APInt &a, bool &castStatus) {
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

/// Fold extension of float constants when there is no information loss due the
/// difference in fp semantics.
OpFoldResult arith::ExtFOp::fold(FoldAdaptor adaptor) {
  if (auto truncFOp = getOperand().getDefiningOp<TruncFOp>()) {
    if (truncFOp.getOperand().getType() == getType()) {
      arith::FastMathFlags truncFMF =
          truncFOp.getFastmath().value_or(arith::FastMathFlags::none);
      bool isTruncContract =
          bitEnumContainsAll(truncFMF, arith::FastMathFlags::contract);
      arith::FastMathFlags extFMF =
          getFastmath().value_or(arith::FastMathFlags::none);
      bool isExtContract =
          bitEnumContainsAll(extFMF, arith::FastMathFlags::contract);
      if (isTruncContract && isExtContract) {
        return truncFOp.getOperand();
      }
    }
  }

  auto resElemType = cast<FloatType>(getElementTypeOrSelf(getType()));
  const llvm::fltSemantics &targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr>(
      adaptor.getOperands(), getType(),
      [&targetSemantics](const APFloat &a, bool &castStatus) {
        FailureOr<APFloat> result = convertFloatValue(a, targetSemantics);
        if (failed(result)) {
          castStatus = false;
          return a;
        }
        return *result;
      });
}

bool arith::ExtFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::greater, FloatType>(inputs, outputs);
}

LogicalResult arith::ExtFOp::verify() { return verifyExtOp<FloatType>(*this); }

//===----------------------------------------------------------------------===//
// ScalingExtFOp
//===----------------------------------------------------------------------===//

bool arith::ScalingExtFOp::areCastCompatible(TypeRange inputs,
                                             TypeRange outputs) {
  return checkWidthChangeCast<std::greater, FloatType>(inputs.front(), outputs);
}

LogicalResult arith::ScalingExtFOp::verify() {
  return verifyExtOp<FloatType>(*this);
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::TruncIOp::fold(FoldAdaptor adaptor) {
  if (matchPattern(getOperand(), m_Op<arith::ExtUIOp>()) ||
      matchPattern(getOperand(), m_Op<arith::ExtSIOp>())) {
    Value src = getOperand().getDefiningOp()->getOperand(0);
    Type srcType = getElementTypeOrSelf(src.getType());
    Type dstType = getElementTypeOrSelf(getType());
    // trunci(zexti(a)) -> trunci(a)
    // trunci(sexti(a)) -> trunci(a)
    if (llvm::cast<IntegerType>(srcType).getWidth() >
        llvm::cast<IntegerType>(dstType).getWidth()) {
      setOperand(src);
      return getResult();
    }

    // trunci(zexti(a)) -> a
    // trunci(sexti(a)) -> a
    if (srcType == dstType)
      return src;
  }

  // trunci(trunci(a)) -> trunci(a))
  if (matchPattern(getOperand(), m_Op<arith::TruncIOp>())) {
    setOperand(getOperand().getDefiningOp()->getOperand(0));
    return getResult();
  }

  Type resType = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      adaptor.getOperands(), getType(),
      [bitWidth](const APInt &a, bool &castStatus) {
        return a.trunc(bitWidth);
      });
}

bool arith::TruncIOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::less, IntegerType>(inputs, outputs);
}

void arith::TruncIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns
      .add<TruncIExtSIToExtSI, TruncIExtUIToExtUI, TruncIShrSIToTrunciShrUI>(
          context);
}

LogicalResult arith::TruncIOp::verify() {
  return verifyTruncateOp<IntegerType>(*this);
}

//===----------------------------------------------------------------------===//
// TruncFOp
//===----------------------------------------------------------------------===//

/// Perform safe const propagation for truncf, i.e., only propagate if FP value
/// can be represented without precision loss.
OpFoldResult arith::TruncFOp::fold(FoldAdaptor adaptor) {
  auto resElemType = cast<FloatType>(getElementTypeOrSelf(getType()));
  if (auto extOp = getOperand().getDefiningOp<arith::ExtFOp>()) {
    Value src = extOp.getIn();
    auto srcType = cast<FloatType>(getElementTypeOrSelf(src.getType()));
    auto intermediateType =
        cast<FloatType>(getElementTypeOrSelf(extOp.getType()));
    // Check if the srcType is representable in the intermediateType.
    if (llvm::APFloatBase::isRepresentableBy(
            srcType.getFloatSemantics(),
            intermediateType.getFloatSemantics())) {
      // truncf(extf(a)) -> truncf(a)
      if (srcType.getWidth() > resElemType.getWidth()) {
        setOperand(src);
        return getResult();
      }

      // truncf(extf(a)) -> a
      if (srcType == resElemType)
        return src;
    }
  }

  const llvm::fltSemantics &targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr>(
      adaptor.getOperands(), getType(),
      [this, &targetSemantics](const APFloat &a, bool &castStatus) {
        RoundingMode roundingMode =
            getRoundingmode().value_or(RoundingMode::to_nearest_even);
        llvm::RoundingMode llvmRoundingMode =
            convertArithRoundingModeToLLVMIR(roundingMode);
        FailureOr<APFloat> result =
            convertFloatValue(a, targetSemantics, llvmRoundingMode);
        if (failed(result)) {
          castStatus = false;
          return a;
        }
        return *result;
      });
}

void arith::TruncFOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<TruncFSIToFPToSIToFP, TruncFUIToFPToUIToFP>(context);
}

bool arith::TruncFOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  return checkWidthChangeCast<std::less, FloatType>(inputs, outputs);
}

LogicalResult arith::TruncFOp::verify() {
  return verifyTruncateOp<FloatType>(*this);
}

//===----------------------------------------------------------------------===//
// ScalingTruncFOp
//===----------------------------------------------------------------------===//

bool arith::ScalingTruncFOp::areCastCompatible(TypeRange inputs,
                                               TypeRange outputs) {
  return checkWidthChangeCast<std::less, FloatType>(inputs.front(), outputs);
}

LogicalResult arith::ScalingTruncFOp::verify() {
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

OpFoldResult arith::UIToFPOp::fold(FoldAdaptor adaptor) {
  Type resEleType = getElementTypeOrSelf(getType());
  return constFoldCastOp<IntegerAttr, FloatAttr>(
      adaptor.getOperands(), getType(),
      [&resEleType](const APInt &a, bool &castStatus) {
        FloatType floatTy = llvm::cast<FloatType>(resEleType);
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

OpFoldResult arith::SIToFPOp::fold(FoldAdaptor adaptor) {
  Type resEleType = getElementTypeOrSelf(getType());
  return constFoldCastOp<IntegerAttr, FloatAttr>(
      adaptor.getOperands(), getType(),
      [&resEleType](const APInt &a, bool &castStatus) {
        FloatType floatTy = llvm::cast<FloatType>(resEleType);
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

OpFoldResult arith::FPToUIOp::fold(FoldAdaptor adaptor) {
  Type resType = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      adaptor.getOperands(), getType(),
      [&bitWidth](const APFloat &a, bool &castStatus) {
        bool ignored;
        APSInt api(bitWidth, /*isUnsigned=*/true);
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

OpFoldResult arith::FPToSIOp::fold(FoldAdaptor adaptor) {
  Type resType = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<FloatAttr, IntegerAttr>(
      adaptor.getOperands(), getType(),
      [&bitWidth](const APFloat &a, bool &castStatus) {
        bool ignored;
        APSInt api(bitWidth, /*isUnsigned=*/false);
        castStatus = APFloat::opInvalidOp !=
                     a.convertToInteger(api, APFloat::rmTowardZero, &ignored);
        return api;
      });
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

static bool areIndexCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(outputs.front());
  if (!srcType || !dstType)
    return false;

  return (srcType.isIndex() && dstType.isSignlessInteger()) ||
         (srcType.isSignlessInteger() && dstType.isIndex());
}

bool arith::IndexCastOp::areCastCompatible(TypeRange inputs,
                                           TypeRange outputs) {
  return areIndexCastCompatible(inputs, outputs);
}

OpFoldResult arith::IndexCastOp::fold(FoldAdaptor adaptor) {
  // index_cast(constant) -> constant
  unsigned resultBitwidth = 64; // Default for index integer attributes.
  if (auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(getType())))
    resultBitwidth = intTy.getWidth();

  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      adaptor.getOperands(), getType(),
      [resultBitwidth](const APInt &a, bool & /*castStatus*/) {
        return a.sextOrTrunc(resultBitwidth);
      });
}

void arith::IndexCastOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<IndexCastOfIndexCast, IndexCastOfExtSI>(context);
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

bool arith::IndexCastUIOp::areCastCompatible(TypeRange inputs,
                                             TypeRange outputs) {
  return areIndexCastCompatible(inputs, outputs);
}

OpFoldResult arith::IndexCastUIOp::fold(FoldAdaptor adaptor) {
  // index_castui(constant) -> constant
  unsigned resultBitwidth = 64; // Default for index integer attributes.
  if (auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(getType())))
    resultBitwidth = intTy.getWidth();

  return constFoldCastOp<IntegerAttr, IntegerAttr>(
      adaptor.getOperands(), getType(),
      [resultBitwidth](const APInt &a, bool & /*castStatus*/) {
        return a.zextOrTrunc(resultBitwidth);
      });
}

void arith::IndexCastUIOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<IndexCastUIOfIndexCastUI, IndexCastUIOfExtUI>(context);
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

bool arith::BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (!areValidCastInputsAndOutputs(inputs, outputs))
    return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, FloatType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, FloatType>(outputs.front());
  if (!srcType || !dstType)
    return false;

  return srcType.getIntOrFloatBitWidth() == dstType.getIntOrFloatBitWidth();
}

OpFoldResult arith::BitcastOp::fold(FoldAdaptor adaptor) {
  auto resType = getType();
  auto operand = adaptor.getIn();
  if (!operand)
    return {};

  /// Bitcast dense elements.
  if (auto denseAttr = llvm::dyn_cast_or_null<DenseElementsAttr>(operand))
    return denseAttr.bitcast(llvm::cast<ShapedType>(resType).getElementType());
  /// Other shaped types unhandled.
  if (llvm::isa<ShapedType>(resType))
    return {};

  /// Bitcast poison.
  if (llvm::isa<ub::PoisonAttr>(operand))
    return ub::PoisonAttr::get(getContext());

  /// Bitcast integer or float to integer or float.
  APInt bits = llvm::isa<FloatAttr>(operand)
                   ? llvm::cast<FloatAttr>(operand).getValue().bitcastToAPInt()
                   : llvm::cast<IntegerAttr>(operand).getValue();
  assert(resType.getIntOrFloatBitWidth() == bits.getBitWidth() &&
         "trying to fold on broken IR: operands have incompatible types");

  if (auto resFloatType = llvm::dyn_cast<FloatType>(resType))
    return FloatAttr::get(resType,
                          APFloat(resFloatType.getFloatSemantics(), bits));
  return IntegerAttr::get(resType, bits);
}

void arith::BitcastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<BitcastOfBitcast>(context);
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

static std::optional<int64_t> getIntegerWidth(Type t) {
  if (auto intType = llvm::dyn_cast<IntegerType>(t)) {
    return intType.getWidth();
  }
  if (auto vectorIntType = llvm::dyn_cast<VectorType>(t)) {
    return llvm::cast<IntegerType>(vectorIntType.getElementType()).getWidth();
  }
  return std::nullopt;
}

OpFoldResult arith::CmpIOp::fold(FoldAdaptor adaptor) {
  // cmpi(pred, x, x)
  if (getLhs() == getRhs()) {
    auto val = applyCmpPredicateToEqualOperands(getPredicate());
    return getBoolAttribute(getType(), val);
  }

  if (matchPattern(adaptor.getRhs(), m_Zero())) {
    if (auto extOp = getLhs().getDefiningOp<ExtSIOp>()) {
      // extsi(%x : i1 -> iN) != 0  ->  %x
      std::optional<int64_t> integerWidth =
          getIntegerWidth(extOp.getOperand().getType());
      if (integerWidth && integerWidth.value() == 1 &&
          getPredicate() == arith::CmpIPredicate::ne)
        return extOp.getOperand();
    }
    if (auto extOp = getLhs().getDefiningOp<ExtUIOp>()) {
      // extui(%x : i1 -> iN) != 0  ->  %x
      std::optional<int64_t> integerWidth =
          getIntegerWidth(extOp.getOperand().getType());
      if (integerWidth && integerWidth.value() == 1 &&
          getPredicate() == arith::CmpIPredicate::ne)
        return extOp.getOperand();
    }

    // arith.cmpi ne, %val, %zero : i1 -> %val
    if (getElementTypeOrSelf(getLhs().getType()).isInteger(1) &&
        getPredicate() == arith::CmpIPredicate::ne)
      return getLhs();
  }

  if (matchPattern(adaptor.getRhs(), m_One())) {
    // arith.cmpi eq, %val, %one : i1 -> %val
    if (getElementTypeOrSelf(getLhs().getType()).isInteger(1) &&
        getPredicate() == arith::CmpIPredicate::eq)
      return getLhs();
  }

  // Move constant to the right side.
  if (adaptor.getLhs() && !adaptor.getRhs()) {
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
        setPredicate(pred.second);
        Value lhs = getLhs();
        Value rhs = getRhs();
        getLhsMutable().assign(rhs);
        getRhsMutable().assign(lhs);
        return getResult();
      }
    }
    llvm_unreachable("unknown cmpi predicate kind");
  }

  // We are moving constants to the right side; So if lhs is constant rhs is
  // guaranteed to be a constant.
  if (auto lhs = llvm::dyn_cast_if_present<TypedAttr>(adaptor.getLhs())) {
    return constFoldBinaryOp<IntegerAttr>(
        adaptor.getOperands(), getI1SameShape(lhs.getType()),
        [pred = getPredicate()](const APInt &lhs, const APInt &rhs) {
          return APInt(1,
                       static_cast<int64_t>(applyCmpPredicate(pred, lhs, rhs)));
        });
  }

  return {};
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

OpFoldResult arith::CmpFOp::fold(FoldAdaptor adaptor) {
  auto lhs = llvm::dyn_cast_if_present<FloatAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_if_present<FloatAttr>(adaptor.getRhs());

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
    FloatType floatTy = llvm::cast<FloatType>(op.getRhs().getType());
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
    auto intTy = llvm::cast<IntegerType>(intVal.getType());
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

//  select %arg, %c1, %c0 => extui %arg
struct SelectToExtUI : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    // Cannot extui i1 to i1, or i1 to f32
    if (!llvm::isa<IntegerType>(op.getType()) || op.getType().isInteger(1))
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
                  op.getLoc(), op.getCondition().getType(), 1)));
      return success();
    }

    return failure();
  }
};

void arith::SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<RedundantSelectFalse, RedundantSelectTrue, SelectNotCond,
              SelectI1ToNot, SelectToExtUI>(context);
}

OpFoldResult arith::SelectOp::fold(FoldAdaptor adaptor) {
  Value trueVal = getTrueValue();
  Value falseVal = getFalseValue();
  if (trueVal == falseVal)
    return trueVal;

  Value condition = getCondition();

  // select true, %0, %1 => %0
  if (matchPattern(adaptor.getCondition(), m_One()))
    return trueVal;

  // select false, %0, %1 => %1
  if (matchPattern(adaptor.getCondition(), m_Zero()))
    return falseVal;

  // If either operand is fully poisoned, return the other.
  if (isa_and_nonnull<ub::PoisonAttr>(adaptor.getTrueValue()))
    return falseVal;

  if (isa_and_nonnull<ub::PoisonAttr>(adaptor.getFalseValue()))
    return trueVal;

  // select %x, true, false => %x
  if (getType().isSignlessInteger(1) &&
      matchPattern(adaptor.getTrueValue(), m_One()) &&
      matchPattern(adaptor.getFalseValue(), m_Zero()))
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

  // Constant-fold constant operands over non-splat constant condition.
  // select %cst_vec, %cst0, %cst1 => %cst2
  if (auto cond =
          llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getCondition())) {
    if (auto lhs =
            llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getTrueValue())) {
      if (auto rhs =
              llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getFalseValue())) {
        SmallVector<Attribute> results;
        results.reserve(static_cast<size_t>(cond.getNumElements()));
        auto condVals = llvm::make_range(cond.value_begin<BoolAttr>(),
                                         cond.value_end<BoolAttr>());
        auto lhsVals = llvm::make_range(lhs.value_begin<Attribute>(),
                                        lhs.value_end<Attribute>());
        auto rhsVals = llvm::make_range(rhs.value_begin<Attribute>(),
                                        rhs.value_end<Attribute>());

        for (auto [condVal, lhsVal, rhsVal] :
             llvm::zip_equal(condVals, lhsVals, rhsVals))
          results.push_back(condVal.getValue() ? lhsVal : rhsVal);

        return DenseElementsAttr::get(lhs.getType(), results);
      }
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
  if (ShapedType condType =
          llvm::dyn_cast<ShapedType>(getCondition().getType()))
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
  if (!llvm::isa<TensorType, VectorType>(resultType))
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

OpFoldResult arith::ShLIOp::fold(FoldAdaptor adaptor) {
  // shli(x, 0) -> x
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();
  // Don't fold if shifting more or equal than the bit width.
  bool bounded = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        bounded = b.ult(b.getBitWidth());
        return a.shl(b);
      });
  return bounded ? result : Attribute();
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ShRUIOp::fold(FoldAdaptor adaptor) {
  // shrui(x, 0) -> x
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();
  // Don't fold if shifting more or equal than the bit width.
  bool bounded = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        bounded = b.ult(b.getBitWidth());
        return a.lshr(b);
      });
  return bounded ? result : Attribute();
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

OpFoldResult arith::ShRSIOp::fold(FoldAdaptor adaptor) {
  // shrsi(x, 0) -> x
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();
  // Don't fold if shifting more or equal than the bit width.
  bool bounded = false;
  auto result = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), [&](const APInt &a, const APInt &b) {
        bounded = b.ult(b.getBitWidth());
        return a.ashr(b);
      });
  return bounded ? result : Attribute();
}

//===----------------------------------------------------------------------===//
// Atomic Enum
//===----------------------------------------------------------------------===//

/// Returns the identity value attribute associated with an AtomicRMWKind op.
TypedAttr mlir::arith::getIdentityValueAttr(AtomicRMWKind kind, Type resultType,
                                            OpBuilder &builder, Location loc,
                                            bool useOnlyFiniteValue) {
  switch (kind) {
  case AtomicRMWKind::maximumf: {
    const llvm::fltSemantics &semantic =
        llvm::cast<FloatType>(resultType).getFloatSemantics();
    APFloat identity = useOnlyFiniteValue
                           ? APFloat::getLargest(semantic, /*Negative=*/true)
                           : APFloat::getInf(semantic, /*Negative=*/true);
    return builder.getFloatAttr(resultType, identity);
  }
  case AtomicRMWKind::maxnumf: {
    const llvm::fltSemantics &semantic =
        llvm::cast<FloatType>(resultType).getFloatSemantics();
    APFloat identity = APFloat::getNaN(semantic, /*Negative=*/true);
    return builder.getFloatAttr(resultType, identity);
  }
  case AtomicRMWKind::addf:
  case AtomicRMWKind::addi:
  case AtomicRMWKind::maxu:
  case AtomicRMWKind::ori:
    return builder.getZeroAttr(resultType);
  case AtomicRMWKind::andi:
    return builder.getIntegerAttr(
        resultType,
        APInt::getAllOnes(llvm::cast<IntegerType>(resultType).getWidth()));
  case AtomicRMWKind::maxs:
    return builder.getIntegerAttr(
        resultType, APInt::getSignedMinValue(
                        llvm::cast<IntegerType>(resultType).getWidth()));
  case AtomicRMWKind::minimumf: {
    const llvm::fltSemantics &semantic =
        llvm::cast<FloatType>(resultType).getFloatSemantics();
    APFloat identity = useOnlyFiniteValue
                           ? APFloat::getLargest(semantic, /*Negative=*/false)
                           : APFloat::getInf(semantic, /*Negative=*/false);

    return builder.getFloatAttr(resultType, identity);
  }
  case AtomicRMWKind::minnumf: {
    const llvm::fltSemantics &semantic =
        llvm::cast<FloatType>(resultType).getFloatSemantics();
    APFloat identity = APFloat::getNaN(semantic, /*Negative=*/false);
    return builder.getFloatAttr(resultType, identity);
  }
  case AtomicRMWKind::mins:
    return builder.getIntegerAttr(
        resultType, APInt::getSignedMaxValue(
                        llvm::cast<IntegerType>(resultType).getWidth()));
  case AtomicRMWKind::minu:
    return builder.getIntegerAttr(
        resultType,
        APInt::getMaxValue(llvm::cast<IntegerType>(resultType).getWidth()));
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

/// Return the identity numeric value associated to the give op.
std::optional<TypedAttr> mlir::arith::getNeutralElement(Operation *op) {
  std::optional<AtomicRMWKind> maybeKind =
      llvm::TypeSwitch<Operation *, std::optional<AtomicRMWKind>>(op)
          // Floating-point operations.
          .Case([](arith::AddFOp op) { return AtomicRMWKind::addf; })
          .Case([](arith::MulFOp op) { return AtomicRMWKind::mulf; })
          .Case([](arith::MaximumFOp op) { return AtomicRMWKind::maximumf; })
          .Case([](arith::MinimumFOp op) { return AtomicRMWKind::minimumf; })
          .Case([](arith::MaxNumFOp op) { return AtomicRMWKind::maxnumf; })
          .Case([](arith::MinNumFOp op) { return AtomicRMWKind::minnumf; })
          // Integer operations.
          .Case([](arith::AddIOp op) { return AtomicRMWKind::addi; })
          .Case([](arith::OrIOp op) { return AtomicRMWKind::ori; })
          .Case([](arith::XOrIOp op) { return AtomicRMWKind::ori; })
          .Case([](arith::AndIOp op) { return AtomicRMWKind::andi; })
          .Case([](arith::MaxUIOp op) { return AtomicRMWKind::maxu; })
          .Case([](arith::MinUIOp op) { return AtomicRMWKind::minu; })
          .Case([](arith::MaxSIOp op) { return AtomicRMWKind::maxs; })
          .Case([](arith::MinSIOp op) { return AtomicRMWKind::mins; })
          .Case([](arith::MulIOp op) { return AtomicRMWKind::muli; })
          .Default([](Operation *op) { return std::nullopt; });
  if (!maybeKind) {
    return std::nullopt;
  }

  bool useOnlyFiniteValue = false;
  auto fmfOpInterface = dyn_cast<ArithFastMathInterface>(op);
  if (fmfOpInterface) {
    arith::FastMathFlagsAttr fmfAttr = fmfOpInterface.getFastMathFlagsAttr();
    useOnlyFiniteValue =
        bitEnumContainsAny(fmfAttr.getValue(), arith::FastMathFlags::ninf);
  }

  // Builder only used as helper for attribute creation.
  OpBuilder b(op->getContext());
  Type resultType = op->getResult(0).getType();

  return getIdentityValueAttr(*maybeKind, resultType, b, op->getLoc(),
                              useOnlyFiniteValue);
}

/// Returns the identity value associated with an AtomicRMWKind op.
Value mlir::arith::getIdentityValue(AtomicRMWKind op, Type resultType,
                                    OpBuilder &builder, Location loc,
                                    bool useOnlyFiniteValue) {
  auto attr =
      getIdentityValueAttr(op, resultType, builder, loc, useOnlyFiniteValue);
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
  case AtomicRMWKind::maximumf:
    return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
  case AtomicRMWKind::minimumf:
    return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
   case AtomicRMWKind::maxnumf:
    return builder.create<arith::MaxNumFOp>(loc, lhs, rhs);
  case AtomicRMWKind::minnumf:
    return builder.create<arith::MinNumFOp>(loc, lhs, rhs);
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
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/ArithOpsEnums.cpp.inc"
