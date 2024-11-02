//===- Types.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/AST/Types.h"
#include "TypeDetail.h"
#include "mlir/Tools/PDLL/AST/Context.h"

using namespace mlir;
using namespace mlir::pdll;
using namespace mlir::pdll::ast;

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::AttributeTypeStorage)
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::ConstraintTypeStorage)
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::OperationTypeStorage)
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::RangeTypeStorage)
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::RewriteTypeStorage)
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::TupleTypeStorage)
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::TypeTypeStorage)
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::pdll::ast::detail::ValueTypeStorage)

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

TypeID Type::getTypeID() const { return impl->typeID; }

Type Type::refineWith(Type other) const {
  if (*this == other)
    return *this;

  // Operation types are compatible if the operation names don't conflict.
  if (auto opTy = dyn_cast<OperationType>()) {
    auto otherOpTy = other.dyn_cast<ast::OperationType>();
    if (!otherOpTy)
      return nullptr;
    if (!otherOpTy.getName())
      return *this;
    if (!opTy.getName())
      return other;

    return nullptr;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// AttributeType
//===----------------------------------------------------------------------===//

AttributeType AttributeType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}

//===----------------------------------------------------------------------===//
// ConstraintType
//===----------------------------------------------------------------------===//

ConstraintType ConstraintType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}

//===----------------------------------------------------------------------===//
// OperationType
//===----------------------------------------------------------------------===//

OperationType OperationType::get(Context &context, Optional<StringRef> name,
                                 const ods::Operation *odsOp) {
  return context.getTypeUniquer().get<ImplTy>(
      /*initFn=*/function_ref<void(ImplTy *)>(),
      std::make_pair(name.getValueOr(""), odsOp));
}

Optional<StringRef> OperationType::getName() const {
  StringRef name = getImplAs<ImplTy>()->getValue().first;
  return name.empty() ? Optional<StringRef>() : Optional<StringRef>(name);
}

const ods::Operation *OperationType::getODSOperation() const {
  return getImplAs<ImplTy>()->getValue().second;
}

//===----------------------------------------------------------------------===//
// RangeType
//===----------------------------------------------------------------------===//

RangeType RangeType::get(Context &context, Type elementType) {
  return context.getTypeUniquer().get<ImplTy>(
      /*initFn=*/function_ref<void(ImplTy *)>(), elementType);
}

Type RangeType::getElementType() const {
  return getImplAs<ImplTy>()->getValue();
}

//===----------------------------------------------------------------------===//
// TypeRangeType

bool TypeRangeType::classof(Type type) {
  RangeType range = type.dyn_cast<RangeType>();
  return range && range.getElementType().isa<TypeType>();
}

TypeRangeType TypeRangeType::get(Context &context) {
  return RangeType::get(context, TypeType::get(context)).cast<TypeRangeType>();
}

//===----------------------------------------------------------------------===//
// ValueRangeType

bool ValueRangeType::classof(Type type) {
  RangeType range = type.dyn_cast<RangeType>();
  return range && range.getElementType().isa<ValueType>();
}

ValueRangeType ValueRangeType::get(Context &context) {
  return RangeType::get(context, ValueType::get(context))
      .cast<ValueRangeType>();
}

//===----------------------------------------------------------------------===//
// RewriteType
//===----------------------------------------------------------------------===//

RewriteType RewriteType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

TupleType TupleType::get(Context &context, ArrayRef<Type> elementTypes,
                         ArrayRef<StringRef> elementNames) {
  assert(elementTypes.size() == elementNames.size());
  return context.getTypeUniquer().get<ImplTy>(
      /*initFn=*/function_ref<void(ImplTy *)>(), elementTypes, elementNames);
}
TupleType TupleType::get(Context &context, ArrayRef<Type> elementTypes) {
  SmallVector<StringRef> elementNames(elementTypes.size());
  return get(context, elementTypes, elementNames);
}

ArrayRef<Type> TupleType::getElementTypes() const {
  return getImplAs<ImplTy>()->getValue().first;
}

ArrayRef<StringRef> TupleType::getElementNames() const {
  return getImplAs<ImplTy>()->getValue().second;
}

//===----------------------------------------------------------------------===//
// TypeType
//===----------------------------------------------------------------------===//

TypeType TypeType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}

//===----------------------------------------------------------------------===//
// ValueType
//===----------------------------------------------------------------------===//

ValueType ValueType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}
