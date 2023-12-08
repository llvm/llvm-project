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
// AbstractAttribute
//===----------------------------------------------------------------------===//

void AbstractAttribute::walkImmediateSubElements(
    Attribute attr, function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkImmediateSubElementsFn(attr, walkAttrsFn, walkTypesFn);
}

Attribute
AbstractAttribute::replaceImmediateSubElements(Attribute attr,
                                               ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  return replaceImmediateSubElementsFn(attr, replAttrs, replTypes);
}

//===----------------------------------------------------------------------===//
// Attribute
//===----------------------------------------------------------------------===//

/// Return the context this attribute belongs to.
MLIRContext *Attribute::getContext() const { return getDialect().getContext(); }

//===----------------------------------------------------------------------===//
// NamedAttribute
//===----------------------------------------------------------------------===//

NamedAttribute::NamedAttribute(StringAttr name, Attribute value)
    : name(name), value(value) {
  assert(name && value && "expected valid attribute name and value");
  assert(!name.empty() && "expected valid attribute name");
}

StringAttr NamedAttribute::getName() const {
  return llvm::cast<StringAttr>(name);
}

Dialect *NamedAttribute::getNameDialect() const {
  return getName().getReferencedDialect();
}

void NamedAttribute::setName(StringAttr newName) {
  assert(name && "expected valid attribute name");
  name = newName;
}

bool NamedAttribute::operator<(const NamedAttribute &rhs) const {
  return getName().compare(rhs.getName()) < 0;
}

bool NamedAttribute::operator<(StringRef rhs) const {
  return getName().getValue().compare(rhs) < 0;
}
