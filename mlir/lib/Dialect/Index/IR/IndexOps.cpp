//===- IndexOps.cpp - Index operation definitions --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace mlir::index;

//===----------------------------------------------------------------------===//
// IndexDialect
//===----------------------------------------------------------------------===//

void IndexDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Index/IR/IndexOps.cpp.inc"
      >();
}

Operation *IndexDialect::materializeConstant(OpBuilder &b, Attribute value,
                                             Type type, Location loc) {
  // Materialize bool constants as `i1`.
  if (auto boolValue = dyn_cast<BoolAttr>(value)) {
    if (!type.isSignlessInteger(1))
      return nullptr;
    return b.create<BoolConstantOp>(loc, type, boolValue);
  }

  // Materialize integer attributes as `index`.
  if (auto indexValue = dyn_cast<IntegerAttr>(value)) {
    if (!indexValue.getType().isa<IndexType>() || !type.isa<IndexType>())
      return nullptr;
    assert(indexValue.getValue().getBitWidth() ==
           IndexType::kInternalStorageBitWidth);
    return b.create<ConstantOp>(loc, indexValue);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Fold Utilities
//===----------------------------------------------------------------------===//

/// Fold an index operation irrespective of the target bitwidth. The
/// operation must satisfy the property:
///
/// ```
/// trunc(f(a, b)) = f(trunc(a), trunc(b))
/// ```
///
/// For all values of `a` and `b`. The function accepts a lambda that computes
/// the integer result, which in turn must satisfy the above property.
static OpFoldResult foldBinaryOpUnchecked(
    ArrayRef<Attribute> operands,
    function_ref<Optional<APInt>(const APInt &, const APInt &)> calculate) {
  assert(operands.size() == 2 && "binary operation expected 2 operands");
  auto lhs = dyn_cast_if_present<IntegerAttr>(operands[0]);
  auto rhs = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!lhs || !rhs)
    return {};

  Optional<APInt> result = calculate(lhs.getValue(), rhs.getValue());
  if (!result)
    return {};
  assert(result->trunc(32) ==
         calculate(lhs.getValue().trunc(32), rhs.getValue().trunc(32)));
  return IntegerAttr::get(IndexType::get(lhs.getContext()), *result);
}

/// Fold an index operation only if the truncated 64-bit result matches the
/// 32-bit result for operations that don't satisfy the above property. These
/// are operations where the upper bits of the operands can affect the lower
/// bits of the results.
///
/// The function accepts a lambda that computes the integer result in both
/// 64-bit and 32-bit. If either call returns `std::nullopt`, the operation is
/// not folded.
static OpFoldResult foldBinaryOpChecked(
    ArrayRef<Attribute> operands,
    function_ref<Optional<APInt>(const APInt &, const APInt &lhs)> calculate) {
  assert(operands.size() == 2 && "binary operation expected 2 operands");
  auto lhs = dyn_cast_if_present<IntegerAttr>(operands[0]);
  auto rhs = dyn_cast_if_present<IntegerAttr>(operands[1]);
  // Only fold index operands.
  if (!lhs || !rhs)
    return {};

  // Compute the 64-bit result and the 32-bit result.
  Optional<APInt> result64 = calculate(lhs.getValue(), rhs.getValue());
  if (!result64)
    return {};
  Optional<APInt> result32 =
      calculate(lhs.getValue().trunc(32), rhs.getValue().trunc(32));
  if (!result32)
    return {};
  // Compare the truncated 64-bit result to the 32-bit result.
  if (result64->trunc(32) != *result32)
    return {};
  // The operation can be folded for these particular operands.
  return IntegerAttr::get(IndexType::get(lhs.getContext()), *result64);
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpUnchecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs + rhs; });
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpUnchecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs - rhs; });
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpUnchecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs * rhs; });
}

//===----------------------------------------------------------------------===//
// DivSOp
//===----------------------------------------------------------------------===//

OpFoldResult DivSOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) -> Optional<APInt> {
        // Don't fold division by zero.
        if (rhs.isZero())
          return std::nullopt;
        return lhs.sdiv(rhs);
      });
}

//===----------------------------------------------------------------------===//
// DivUOp
//===----------------------------------------------------------------------===//

OpFoldResult DivUOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) -> Optional<APInt> {
        // Don't fold division by zero.
        if (rhs.isZero())
          return std::nullopt;
        return lhs.udiv(rhs);
      });
}

//===----------------------------------------------------------------------===//
// CeilDivSOp
//===----------------------------------------------------------------------===//

/// Compute `ceildivs(n, m)` as `x = m > 0 ? -1 : 1` and then
/// `n*m > 0 ? (n+x)/m + 1 : -(-n/m)`.
static Optional<APInt> calculateCeilDivS(const APInt &n, const APInt &m) {
  // Don't fold division by zero.
  if (m.isZero())
    return std::nullopt;
  // Short-circuit the zero case.
  if (n.isZero())
    return n;

  bool mGtZ = m.sgt(0);
  if (n.sgt(0) != mGtZ) {
    // If the operands have different signs, compute the negative result. Signed
    // division overflow is not possible, since if `m == -1`, `n` can be at most
    // `INT_MAX`, and `-INT_MAX != INT_MIN` in two's complement.
    return -(-n).sdiv(m);
  }
  // Otherwise, compute the positive result. Signed division overflow is not
  // possible since if `m == -1`, `x` will be `1`.
  int64_t x = mGtZ ? -1 : 1;
  return (n + x).sdiv(m) + 1;
}

OpFoldResult CeilDivSOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(adaptor.getOperands(), calculateCeilDivS);
}

//===----------------------------------------------------------------------===//
// CeilDivUOp
//===----------------------------------------------------------------------===//

OpFoldResult CeilDivUOp::fold(FoldAdaptor adaptor) {
  // Compute `ceildivu(n, m)` as `n == 0 ? 0 : (n-1)/m + 1`.
  return foldBinaryOpChecked(
      adaptor.getOperands(),
      [](const APInt &n, const APInt &m) -> Optional<APInt> {
        // Don't fold division by zero.
        if (m.isZero())
          return std::nullopt;
        // Short-circuit the zero case.
        if (n.isZero())
          return n;

        return (n - 1).udiv(m) + 1;
      });
}

//===----------------------------------------------------------------------===//
// FloorDivSOp
//===----------------------------------------------------------------------===//

/// Compute `floordivs(n, m)` as `x = m < 0 ? 1 : -1` and then
/// `n*m < 0 ? -1 - (x-n)/m : n/m`.
static Optional<APInt> calculateFloorDivS(const APInt &n, const APInt &m) {
  // Don't fold division by zero.
  if (m.isZero())
    return std::nullopt;
  // Short-circuit the zero case.
  if (n.isZero())
    return n;

  bool mLtZ = m.slt(0);
  if (n.slt(0) == mLtZ) {
    // If the operands have the same sign, compute the positive result.
    return n.sdiv(m);
  }
  // If the operands have different signs, compute the negative result. Signed
  // division overflow is not possible since if `m == -1`, `x` will be 1 and
  // `n` can be at most `INT_MAX`.
  int64_t x = mLtZ ? 1 : -1;
  return -1 - (x - n).sdiv(m);
}

OpFoldResult FloorDivSOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(adaptor.getOperands(), calculateFloorDivS);
}

//===----------------------------------------------------------------------===//
// RemSOp
//===----------------------------------------------------------------------===//

OpFoldResult RemSOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs.srem(rhs); });
}

//===----------------------------------------------------------------------===//
// RemUOp
//===----------------------------------------------------------------------===//

OpFoldResult RemUOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs.urem(rhs); });
}

//===----------------------------------------------------------------------===//
// MaxSOp
//===----------------------------------------------------------------------===//

OpFoldResult MaxSOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(adaptor.getOperands(),
                             [](const APInt &lhs, const APInt &rhs) {
                               return lhs.sgt(rhs) ? lhs : rhs;
                             });
}

//===----------------------------------------------------------------------===//
// MaxUOp
//===----------------------------------------------------------------------===//

OpFoldResult MaxUOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(adaptor.getOperands(),
                             [](const APInt &lhs, const APInt &rhs) {
                               return lhs.ugt(rhs) ? lhs : rhs;
                             });
}

//===----------------------------------------------------------------------===//
// MinSOp
//===----------------------------------------------------------------------===//

OpFoldResult MinSOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(adaptor.getOperands(), [](const APInt &lhs, const APInt &rhs) {
    return lhs.slt(rhs) ? lhs : rhs;
  });
}

//===----------------------------------------------------------------------===//
// MinUOp
//===----------------------------------------------------------------------===//

OpFoldResult MinUOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(adaptor.getOperands(), [](const APInt &lhs, const APInt &rhs) {
    return lhs.ult(rhs) ? lhs : rhs;
  });
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

OpFoldResult ShlOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpUnchecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) -> Optional<APInt> {
        // We cannot fold if the RHS is greater than or equal to 32 because
        // this would be UB in 32-bit systems but not on 64-bit systems. RHS is
        // already treated as unsigned.
        if (rhs.uge(32))
          return {};
        return lhs << rhs;
      });
}

//===----------------------------------------------------------------------===//
// ShrSOp
//===----------------------------------------------------------------------===//

OpFoldResult ShrSOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) -> Optional<APInt> {
        // Don't fold if RHS is greater than or equal to 32.
        if (rhs.uge(32))
          return {};
        return lhs.ashr(rhs);
      });
}

//===----------------------------------------------------------------------===//
// ShrUOp
//===----------------------------------------------------------------------===//

OpFoldResult ShrUOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpChecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) -> Optional<APInt> {
        // Don't fold if RHS is greater than or equal to 32.
        if (rhs.uge(32))
          return {};
        return lhs.lshr(rhs);
      });
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpUnchecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs & rhs; });
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpUnchecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs | rhs; });
}

//===----------------------------------------------------------------------===//
// XOrOp
//===----------------------------------------------------------------------===//

OpFoldResult XOrOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOpUnchecked(
      adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) { return lhs ^ rhs; });
}

//===----------------------------------------------------------------------===//
// CastSOp
//===----------------------------------------------------------------------===//

bool CastSOp::areCastCompatible(TypeRange lhsTypes, TypeRange rhsTypes) {
  return lhsTypes.front().isa<IndexType>() != rhsTypes.front().isa<IndexType>();
}

//===----------------------------------------------------------------------===//
// CastUOp
//===----------------------------------------------------------------------===//

bool CastUOp::areCastCompatible(TypeRange lhsTypes, TypeRange rhsTypes) {
  return lhsTypes.front().isa<IndexType>() != rhsTypes.front().isa<IndexType>();
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

/// Compare two integers according to the comparison predicate.
bool compareIndices(const APInt &lhs, const APInt &rhs,
                    IndexCmpPredicate pred) {
  switch (pred) {
  case IndexCmpPredicate::EQ:
    return lhs.eq(rhs);
  case IndexCmpPredicate::NE:
    return lhs.ne(rhs);
  case IndexCmpPredicate::SGE:
    return lhs.sge(rhs);
  case IndexCmpPredicate::SGT:
    return lhs.sgt(rhs);
  case IndexCmpPredicate::SLE:
    return lhs.sle(rhs);
  case IndexCmpPredicate::SLT:
    return lhs.slt(rhs);
  case IndexCmpPredicate::UGE:
    return lhs.uge(rhs);
  case IndexCmpPredicate::UGT:
    return lhs.ugt(rhs);
  case IndexCmpPredicate::ULE:
    return lhs.ule(rhs);
  case IndexCmpPredicate::ULT:
    return lhs.ult(rhs);
  }
  llvm_unreachable("unhandled IndexCmpPredicate predicate");
}

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};

  // Perform the comparison in 64-bit and 32-bit.
  bool result64 = compareIndices(lhs.getValue(), rhs.getValue(), getPred());
  bool result32 = compareIndices(lhs.getValue().trunc(32),
                                 rhs.getValue().trunc(32), getPred());
  if (result64 != result32)
    return {};
  return BoolAttr::get(getContext(), result64);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "idx" << getValueAttr().getValue();
  setNameFn(getResult(), specialName.str());
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

void ConstantOp::build(OpBuilder &b, OperationState &state, int64_t value) {
  build(b, state, b.getIndexType(), b.getIndexAttr(value));
}

//===----------------------------------------------------------------------===//
// BoolConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult BoolConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

void BoolConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getValue() ? "true" : "false");
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Index/IR/IndexOps.cpp.inc"
