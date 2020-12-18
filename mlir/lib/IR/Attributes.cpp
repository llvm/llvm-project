//===- Attributes.cpp - MLIR Affine Expr Classes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// AttributeStorage
//===----------------------------------------------------------------------===//

AttributeStorage::AttributeStorage(Type type)
    : type(type.getAsOpaquePointer()) {}
AttributeStorage::AttributeStorage() : type(nullptr) {}

Type AttributeStorage::getType() const {
  return Type::getFromOpaquePointer(type);
}
void AttributeStorage::setType(Type newType) {
  type = newType.getAsOpaquePointer();
}

//===----------------------------------------------------------------------===//
// Attribute
//===----------------------------------------------------------------------===//

/// Return the type of this attribute.
Type Attribute::getType() const { return impl->getType(); }

/// Return the context this attribute belongs to.
MLIRContext *Attribute::getContext() const { return getDialect().getContext(); }

/// Get the dialect this attribute is registered to.
Dialect &Attribute::getDialect() const {
  return impl->getAbstractAttribute().getDialect();
}

//===----------------------------------------------------------------------===//
// NamedAttribute
//===----------------------------------------------------------------------===//

bool mlir::operator<(const NamedAttribute &lhs, const NamedAttribute &rhs) {
  return strcmp(lhs.first.data(), rhs.first.data()) < 0;
}
bool mlir::operator<(const NamedAttribute &lhs, StringRef rhs) {
  // This is correct even when attr.first.data()[name.size()] is not a zero
  // string terminator, because we only care about a less than comparison.
  // This can't use memcmp, because it doesn't guarantee that it will stop
  // reading both buffers if one is shorter than the other, even if there is
  // a difference.
  return strncmp(lhs.first.data(), rhs.data(), rhs.size()) < 0;
}
