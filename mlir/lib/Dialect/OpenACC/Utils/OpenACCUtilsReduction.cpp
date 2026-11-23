//===- OpenACCUtilsReduction.cpp - OpenACC reduction utilities ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsReduction.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace acc {

static bool isFloatOrComplexType(Type ty) {
  return isa<FloatType, ComplexType>(ty);
}

SmallVector<GPUParallelDimAttr>
getReductionCombineParDims(ReductionCombineOp reductionCombineOp) {
  if (GPUParallelDimsAttr parDimsAttr = getParDimsAttr(reductionCombineOp))
    return SmallVector<GPUParallelDimAttr>(parDimsAttr.getArray());
  llvm_unreachable(
      "expected parallel dimensions attribute for reduction combine op");
}

SmallVector<GPUParallelDimAttr>
getReductionCombineParDims(ReductionCombineRegionOp combineRegionOp) {
  for (Operation *user : combineRegionOp.getSrcVar().getUsers()) {
    if (auto accumulateOp = dyn_cast<ReductionAccumulateOp>(user))
      return SmallVector<GPUParallelDimAttr>(
          accumulateOp.getParDims().getArray());
  }
  if (GPUParallelDimsAttr parDimsAttr = getParDimsAttr(combineRegionOp))
    return SmallVector<GPUParallelDimAttr>(parDimsAttr.getArray());
  return {};
}

ReductionOperator translateAtomicRMWKind(arith::AtomicRMWKind kind) {
  switch (kind) {
  case arith::AtomicRMWKind::addf:
  case arith::AtomicRMWKind::addi:
    return ReductionOperator::AccAdd;
  case arith::AtomicRMWKind::mulf:
  case arith::AtomicRMWKind::muli:
    return ReductionOperator::AccMul;
  case arith::AtomicRMWKind::maxs:
  case arith::AtomicRMWKind::maxu:
  case arith::AtomicRMWKind::maximumf:
  case arith::AtomicRMWKind::maxnumf:
    return ReductionOperator::AccMax;
  case arith::AtomicRMWKind::minu:
  case arith::AtomicRMWKind::mins:
  case arith::AtomicRMWKind::minimumf:
  case arith::AtomicRMWKind::minnumf:
    return ReductionOperator::AccMin;
  case arith::AtomicRMWKind::andi:
    return ReductionOperator::AccIand;
  case arith::AtomicRMWKind::ori:
    return ReductionOperator::AccIor;
  case arith::AtomicRMWKind::xori:
    return ReductionOperator::AccXor;
  case arith::AtomicRMWKind::assign:
    break;
  }
  llvm_unreachable("unsupported atomic kind");
}

std::optional<arith::AtomicRMWKind>
translateACCReductionOperator(ReductionOperator redOp, Type type) {
  if (type.isInteger() && type.isUnsignedInteger())
    return std::nullopt;

  if (auto reducible = dyn_cast<ReducibleType>(type)) {
    if (std::optional<arith::AtomicRMWKind> kind =
            reducible.getAtomicRMWKind(redOp))
      return kind;
    return std::nullopt;
  }

  switch (redOp) {
  case ReductionOperator::AccAdd:
    if (type.isInteger())
      return arith::AtomicRMWKind::addi;
    if (isFloatOrComplexType(type))
      return arith::AtomicRMWKind::addf;
    break;
  case ReductionOperator::AccMul:
    if (type.isInteger())
      return arith::AtomicRMWKind::muli;
    if (isFloatOrComplexType(type))
      return arith::AtomicRMWKind::mulf;
    break;
  case ReductionOperator::AccMax:
    if (type.isInteger())
      return arith::AtomicRMWKind::maxs;
    if (type.isFloat())
      return arith::AtomicRMWKind::maxnumf;
    break;
  case ReductionOperator::AccMaximumf:
    return arith::AtomicRMWKind::maximumf;
  case ReductionOperator::AccMaxnumf:
    return arith::AtomicRMWKind::maxnumf;
  case ReductionOperator::AccMin:
    if (type.isInteger())
      return arith::AtomicRMWKind::mins;
    if (type.isFloat())
      return arith::AtomicRMWKind::minnumf;
    break;
  case ReductionOperator::AccMinimumf:
    return arith::AtomicRMWKind::minimumf;
  case ReductionOperator::AccMinnumf:
    return arith::AtomicRMWKind::minnumf;
  case ReductionOperator::AccIand:
  case ReductionOperator::AccLand:
    if (type.isInteger())
      return arith::AtomicRMWKind::andi;
    break;
  case ReductionOperator::AccIor:
  case ReductionOperator::AccLor:
    if (type.isInteger())
      return arith::AtomicRMWKind::ori;
    break;
  case ReductionOperator::AccXor:
  case ReductionOperator::AccNeqv:
    if (type.isInteger())
      return arith::AtomicRMWKind::xori;
    break;
  case ReductionOperator::AccEqv:
  case ReductionOperator::AccNone:
    break;
  }
  return std::nullopt;
}

static TypedAttr getReductionIdentityValueAttr(arith::AtomicRMWKind kind,
                                               Type type, OpBuilder &builder,
                                               Location loc,
                                               bool useOnlyFiniteValue) {
  if (type.isIntOrIndexOrFloat()) {
    TypedAttr attr = arith::getIdentityValueAttr(kind, type, builder, loc,
                                                 useOnlyFiniteValue);
    if (!attr)
      emitError(loc) << "reduction identity: operator not supported " << kind;
    return attr;
  }
  if (auto complexTy = dyn_cast<ComplexType>(type)) {
    auto eltTy = dyn_cast<FloatType>(complexTy.getElementType());
    if (!eltTy) {
      emitError(loc) << "reduction identity: complex with non-floating "
                        "element type";
      return nullptr;
    }
    switch (kind) {
    case arith::AtomicRMWKind::addf: {
      TypedAttr scalarAttr = arith::getIdentityValueAttr(
          kind, eltTy, builder, loc, useOnlyFiniteValue);
      assert(scalarAttr && "expected scalar identity for complex reduction");
      double d = cast<FloatAttr>(scalarAttr).getValue().convertToDouble();
      return complex::NumberAttr::get(complexTy, d, d);
    }
    case arith::AtomicRMWKind::mulf: {
      TypedAttr scalarAttr = arith::getIdentityValueAttr(
          kind, eltTy, builder, loc, useOnlyFiniteValue);
      assert(scalarAttr &&
             "expected scalar identity for complex mulf reduction");
      auto realPart = cast<FloatAttr>(scalarAttr).getValue();
      return complex::NumberAttr::get(complexTy, realPart.convertToDouble(),
                                      0.0);
    }
    default:
      emitError(loc)
          << "reduction identity: operator not supported for complex " << kind;
      return nullptr;
    }
  }
  emitError(loc) << "reduction identity: type not supported " << type;
  return nullptr;
}

Value createIdentityValue(OpBuilder &b, Location loc, Type type,
                          arith::AtomicRMWKind kind, bool useOnlyFiniteValue) {
  TypedAttr typedAttr =
      getReductionIdentityValueAttr(kind, type, b, loc, useOnlyFiniteValue);
  assert(typedAttr && "expected identity attribute");
  if (auto numAttr = dyn_cast<complex::NumberAttr>(typedAttr)) {
    auto complexTy = cast<ComplexType>(numAttr.getType());
    auto floatElt = cast<FloatType>(complexTy.getElementType());
    Value realVal = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(floatElt, numAttr.getReal()));
    Value imagVal = arith::ConstantOp::create(
        b, loc, b.getFloatAttr(floatElt, numAttr.getImag()));
    return complex::CreateOp::create(b, loc, complexTy, realVal, imagVal);
  }
  return arith::ConstantOp::create(b, loc, typedAttr);
}

Value generateReductionOp(OpBuilder &b, Location loc, Value lhs, Value rhs,
                          arith::AtomicRMWKind kind) {
  assert(lhs.getType() == rhs.getType() &&
         "expected same type for lhs and rhs");
  if (isa<ComplexType>(lhs.getType())) {
    switch (kind) {
    case arith::AtomicRMWKind::addf:
      return complex::AddOp::create(b, loc, lhs, rhs);
    case arith::AtomicRMWKind::mulf:
      return complex::MulOp::create(b, loc, lhs, rhs);
    default:
      llvm_unreachable("unsupported complex atomic reduction kind");
    }
  }
  return arith::getReductionOp(kind, b, loc, lhs, rhs);
}

} // namespace acc
} // namespace mlir
