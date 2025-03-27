//===- Type.cpp - Type class ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type wrapper to simplify using TableGen Record defining a MLIR Type.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Type.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::Record;

TypeConstraint::TypeConstraint(const llvm::DefInit *init)
    : TypeConstraint(init->getDef()) {}

bool TypeConstraint::isOptional() const {
  return def->isSubClassOf("Optional");
}

bool TypeConstraint::isVariadic() const {
  return def->isSubClassOf("Variadic");
}

bool TypeConstraint::isVariadicOfVariadic() const {
  return def->isSubClassOf("VariadicOfVariadic");
}

StringRef TypeConstraint::getVariadicOfVariadicSegmentSizeAttr() const {
  assert(isVariadicOfVariadic());
  return def->getValueAsString("segmentAttrName");
}

// Returns the builder call for this constraint if this is a buildable type,
// returns std::nullopt otherwise.
std::optional<StringRef> TypeConstraint::getBuilderCall() const {
  const Record *baseType = def;
  if (isVariableLength())
    baseType = baseType->getValueAsDef("baseType");

  // Check to see if this type constraint has a builder call.
  const llvm::RecordVal *builderCall = baseType->getValue("builderCall");
  if (!builderCall || !builderCall->getValue())
    return std::nullopt;
  return TypeSwitch<const llvm::Init *, std::optional<StringRef>>(
             builderCall->getValue())
      .Case<llvm::StringInit>([&](auto *init) {
        StringRef value = init->getValue();
        return value.empty() ? std::optional<StringRef>() : value;
      })
      .Default([](auto *) { return std::nullopt; });
}

// Return the C++ type for this type (which may just be ::mlir::Type).
StringRef TypeConstraint::getCppType() const {
  return def->getValueAsString("cppType");
}

Type::Type(const Record *record) : TypeConstraint(record) {}

Dialect Type::getDialect() const {
  return Dialect(def->getValueAsDef("dialect"));
}
