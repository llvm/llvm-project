//===- LLVMArithOps.cpp - LLVM dialect arithmetic ops ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Arith` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMArithOps.h"

#include "IR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

/// Returns a boolean type that has the same shape as `type`. It supports both
/// fixed size vectors as well as scalable vectors.
Type mlir::LLVM::getI1SameShape(Type type) {
  Type i1Type = IntegerType::get(type.getContext(), 1);
  if (LLVM::isCompatibleVectorType(type))
    return LLVM::getVectorType(i1Type, LLVM::getVectorNumElements(type));
  return i1Type;
}

//===----------------------------------------------------------------------===//
// Printing, parsing, folding and builder for LLVM::CmpOp.
//===----------------------------------------------------------------------===//

template <typename PredicateAttr, typename Predicate>
static ParseResult parseCmpPredicateImpl(
    OpAsmParser &parser, PredicateAttr &predicate,
    function_ref<std::optional<Predicate>(StringRef)> symbolize) {
  std::string spelling;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseString(&spelling))
    return failure();
  std::optional<Predicate> value = symbolize(spelling);
  if (!value)
    return parser.emitError(loc)
           << "'" << spelling
           << "' is an incorrect value of the 'predicate' attribute";
  predicate = PredicateAttr::get(parser.getContext(), *value);
  return success();
}

ParseResult mlir::LLVM::parseCmpPredicate(OpAsmParser &parser,
                                          ICmpPredicateAttr &predicate) {
  return parseCmpPredicateImpl<ICmpPredicateAttr, ICmpPredicate>(
      parser, predicate,
      [](StringRef spelling) { return symbolizeICmpPredicate(spelling); });
}

ParseResult mlir::LLVM::parseCmpPredicate(OpAsmParser &parser,
                                          FCmpPredicateAttr &predicate) {
  return parseCmpPredicateImpl<FCmpPredicateAttr, FCmpPredicate>(
      parser, predicate,
      [](StringRef spelling) { return symbolizeFCmpPredicate(spelling); });
}

void mlir::LLVM::printCmpPredicate(OpAsmPrinter &printer, Operation *,
                                   ICmpPredicateAttr predicate) {
  printer << '"' << stringifyICmpPredicate(predicate.getValue()) << '"';
}

void mlir::LLVM::printCmpPredicate(OpAsmPrinter &printer, Operation *,
                                   FCmpPredicateAttr predicate) {
  printer << '"' << stringifyFCmpPredicate(predicate.getValue()) << '"';
}

/// Returns a scalar or vector boolean attribute of the given type.
static Attribute getBoolAttribute(Type type, MLIRContext *ctx, bool value) {
  auto boolAttr = BoolAttr::get(ctx, value);
  ShapedType shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return boolAttr;
  return DenseElementsAttr::get(shapedType, boolAttr);
}

OpFoldResult ICmpOp::fold(FoldAdaptor adaptor) {
  if (getPredicate() != ICmpPredicate::eq &&
      getPredicate() != ICmpPredicate::ne)
    return {};

  // cmpi(eq/ne, x, x) -> true/false
  if (getLhs() == getRhs())
    return getBoolAttribute(getType(), getContext(),
                            getPredicate() == ICmpPredicate::eq);

  // cmpi(eq/ne, alloca, null) -> false/true
  if (getLhs().getDefiningOp<AllocaOp>() && getRhs().getDefiningOp<ZeroOp>())
    return getBoolAttribute(getType(), getContext(),
                            getPredicate() == ICmpPredicate::ne);

  // cmpi(eq/ne, null, alloca) -> cmpi(eq/ne, alloca, null)
  if (getLhs().getDefiningOp<ZeroOp>() && getRhs().getDefiningOp<AllocaOp>()) {
    Value lhs = getLhs();
    Value rhs = getRhs();
    getLhsMutable().assign(rhs);
    getRhsMutable().assign(lhs);
    return getResult();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::ShlOp::fold(FoldAdaptor adaptor) {
  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!rhs)
    return {};

  if (rhs.getValue().uge(getLhs().getType().getIntOrFloatBitWidth()))
    return {}; // TODO: Fold into poison.

  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  if (!lhs)
    return {};

  return IntegerAttr::get(getType(), lhs.getValue().shl(rhs.getValue()));
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::OrOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  if (!lhs)
    return {};

  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!rhs)
    return {};

  return IntegerAttr::get(getType(), lhs.getValue() | rhs.getValue());
}

//===----------------------------------------------------------------------===//
// UDivOp
//===----------------------------------------------------------------------===//
Speculation::Speculatability UDivOp::getSpeculatability() {
  // X / 0 => UB
  Value divisor = getRhs();
  if (matchPattern(divisor, m_IntRangeWithoutZeroU()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// SDivOp
//===----------------------------------------------------------------------===//
Speculation::Speculatability SDivOp::getSpeculatability() {
  // This function conservatively assumes that all signed division by -1 are
  // not speculatable.
  // X / 0 => UB
  // INT_MIN / -1 => UB
  Value divisor = getRhs();
  if (matchPattern(divisor, m_IntRangeWithoutZeroS()) &&
      matchPattern(divisor, m_IntRangeWithoutNegOneS()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// URemOp
//===----------------------------------------------------------------------===//

Speculation::Speculatability URemOp::getSpeculatability() {
  // urem(x, 0) => UB
  Value divisor = getRhs();
  if (matchPattern(divisor, m_IntRangeWithoutZeroU()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// SRemOp
//===----------------------------------------------------------------------===//

Speculation::Speculatability SRemOp::getSpeculatability() {
  // srem(x, 0) => UB
  // srem(INT_MIN, -1) => UB
  Value divisor = getRhs();
  if (matchPattern(divisor, m_IntRangeWithoutZeroS()) &&
      matchPattern(divisor, m_IntRangeWithoutNegOneS()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMArithOps.cpp.inc"
