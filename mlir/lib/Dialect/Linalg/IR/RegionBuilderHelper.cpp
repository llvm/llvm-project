//===- RegionBuilderHelper.cpp - Region Builder Helper class    -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of RegionBuilderHelper class.
//
//===----------------------------------------------------------------------===//

#include "RegionBuilderHelper.h"

namespace mlir {
namespace linalg {

Value RegionBuilderHelper::buildUnaryFn(
    UnaryFn unaryFn, Value arg, function_ref<InFlightDiagnostic()> emitError) {
  if (!isFloatingPoint(arg)) {
    if (emitError) {
      emitError() << "unsupported non numeric type";
      return nullptr;
    }
    llvm_unreachable("unsupported non numeric type");
  }

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);
  switch (unaryFn) {
  case UnaryFn::exp:
    return math::ExpOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::log:
    return math::LogOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::abs:
    return math::AbsFOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::ceil:
    return math::CeilOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::floor:
    return math::FloorOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::negf:
    return arith::NegFOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::reciprocal: {
    Attribute oneAttr = builder.getOneAttr(arg.getType());
    auto one = arith::ConstantOp::create(builder, arg.getLoc(),
                                         llvm::cast<TypedAttr>(oneAttr));
    return arith::DivFOp::create(builder, arg.getLoc(), one, arg);
  }
  case UnaryFn::round:
    return math::RoundOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::sqrt:
    return math::SqrtOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::rsqrt:
    return math::RsqrtOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::square:
    return arith::MulFOp::create(builder, arg.getLoc(), arg, arg);
  case UnaryFn::tanh:
    return math::TanhOp::create(builder, arg.getLoc(), arg);
  case UnaryFn::erf:
    return math::ErfOp::create(builder, arg.getLoc(), arg);
  }

  if (emitError) {
    emitError() << "unsupported unary function";
    return nullptr;
  }
  llvm_unreachable("unsupported unary function");
}

Value RegionBuilderHelper::buildBinaryFn(
    BinaryFn binaryFn, Value arg0, Value arg1,
    function_ref<InFlightDiagnostic()> emitError) {

  bool allComplex = isComplex(arg0) && isComplex(arg1);
  bool allFloatingPoint = isFloatingPoint(arg0) && isFloatingPoint(arg1);
  bool allInteger = isInteger(arg0) && isInteger(arg1);
  bool allBool = allInteger && arg0.getType().getIntOrFloatBitWidth() == 1 &&
                 arg1.getType().getIntOrFloatBitWidth() == 1;

  if (!allComplex && !allFloatingPoint && !allInteger) {
    if (emitError) {
      emitError()
          << "Cannot build binary Linalg operation: expects allComplex, "
             "allFloatingPoint, or allInteger, got "
          << arg0.getType() << " and " << arg1.getType();
      return nullptr;
    }
    llvm_unreachable("unsupported non numeric type");
  }

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);
  switch (binaryFn) {
  case BinaryFn::add:
    if (allComplex)
      return complex::AddOp::create(builder, arg0.getLoc(), arg0, arg1);
    if (allFloatingPoint)
      return arith::AddFOp::create(builder, arg0.getLoc(), arg0, arg1);
    if (allBool)
      return arith::OrIOp::create(builder, arg0.getLoc(), arg0, arg1);
    return arith::AddIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::sub:
    if (allComplex)
      return complex::SubOp::create(builder, arg0.getLoc(), arg0, arg1);
    if (allFloatingPoint)
      return arith::SubFOp::create(builder, arg0.getLoc(), arg0, arg1);
    if (allBool) {
      if (emitError) {
        emitError() << "unsupported operation: sub with bools";
        return nullptr;
      }
      llvm_unreachable("unsupported operation: sub with bools");
    }
    return arith::SubIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::mul:
    if (allComplex)
      return complex::MulOp::create(builder, arg0.getLoc(), arg0, arg1);
    if (allFloatingPoint)
      return arith::MulFOp::create(builder, arg0.getLoc(), arg0, arg1);
    if (allBool)
      return arith::AndIOp::create(builder, arg0.getLoc(), arg0, arg1);
    return arith::MulIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::div:
    if (allComplex)
      return complex::DivOp::create(builder, arg0.getLoc(), arg0, arg1);
    if (allFloatingPoint)
      return arith::DivFOp::create(builder, arg0.getLoc(), arg0, arg1);
    if (allBool) {
      if (emitError) {
        emitError() << "unsupported operation: div with bools";
        return nullptr;
      }
      llvm_unreachable("unsupported operation: div with bools");
    }
    return arith::DivSIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::div_unsigned:
    if (!allInteger || allBool) {
      if (emitError) {
        emitError() << "unsupported operation: unsigned div not on uint";
        return nullptr;
      }
      llvm_unreachable("unsupported operation: unsigned div not on uint");
    }
    return arith::DivUIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::max_signed:
    assert(!allComplex);
    if (allFloatingPoint)
      return arith::MaximumFOp::create(builder, arg0.getLoc(), arg0, arg1);
    return arith::MaxSIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::min_signed:
    assert(!allComplex);
    if (allFloatingPoint)
      return arith::MinimumFOp::create(builder, arg0.getLoc(), arg0, arg1);
    return arith::MinSIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::max_unsigned:
    assert(!allComplex);
    if (allFloatingPoint)
      return arith::MaximumFOp::create(builder, arg0.getLoc(), arg0, arg1);
    return arith::MaxUIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::min_unsigned:
    assert(!allComplex);
    if (allFloatingPoint)
      return arith::MinimumFOp::create(builder, arg0.getLoc(), arg0, arg1);
    return arith::MinUIOp::create(builder, arg0.getLoc(), arg0, arg1);
  case BinaryFn::powf:
    assert(allFloatingPoint);
    return math::PowFOp::create(builder, arg0.getLoc(), arg0, arg1);
  }

  if (emitError) {
    emitError() << "unsupported binary function";
    return nullptr;
  }
  llvm_unreachable("unsupported binary function");
}

Value RegionBuilderHelper::buildTernaryFn(
    TernaryFn ternaryFn, Value arg0, Value arg1, Value arg2,
    function_ref<InFlightDiagnostic()> emitError) {
  bool headBool =
      isInteger(arg0) && arg0.getType().getIntOrFloatBitWidth() == 1;
  bool tailFloatingPoint =
      isFloatingPoint(arg0) && isFloatingPoint(arg1) && isFloatingPoint(arg2);
  bool tailInteger = isInteger(arg0) && isInteger(arg1) && isInteger(arg2);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);

  switch (ternaryFn) {
  case TernaryFn::select:
    if (!headBool && !(tailFloatingPoint || tailInteger))
      llvm_unreachable("unsupported non numeric type");
    return arith::SelectOp::create(builder, arg0.getLoc(), arg0, arg1, arg2);
  }

  if (emitError) {
    emitError() << "unsupported ternary function";
    return nullptr;
  }
  llvm_unreachable("unsupported ternary function");
}

Value RegionBuilderHelper::buildTypeFn(
    TypeFn typeFn, Type toType, Value operand,
    function_ref<InFlightDiagnostic()> emitError) {
  switch (typeFn) {
  case TypeFn::cast_signed:
    return cast(toType, operand, false);
  case TypeFn::cast_unsigned:
    return cast(toType, operand, true);
  }

  if (emitError) {
    emitError() << "unsupported type conversion function";
    return nullptr;
  }
  llvm_unreachable("unsupported type conversion function");
}

void RegionBuilderHelper::yieldOutputs(ValueRange values) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);
  Location loc = builder.getUnknownLoc();
  YieldOp::create(builder, loc, values);
}

Value RegionBuilderHelper::constant(const std::string &value) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&block);
  Location loc = builder.getUnknownLoc();
  Attribute valueAttr = parseAttribute(value, builder.getContext());
  return arith::ConstantOp::create(builder, loc,
                                   llvm::cast<TypedAttr>(valueAttr));
}

} // namespace linalg

} // namespace mlir
