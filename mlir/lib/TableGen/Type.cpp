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
// returns None otherwise.
Optional<StringRef> TypeConstraint::getBuilderCall() const {
  const llvm::Record *baseType = def;
  if (isVariableLength())
    baseType = baseType->getValueAsDef("baseType");

  // Check to see if this type constraint has a builder call.
  const llvm::RecordVal *builderCall = baseType->getValue("builderCall");
  if (!builderCall || !builderCall->getValue())
    return std::nullopt;
  return TypeSwitch<llvm::Init *, Optional<StringRef>>(builderCall->getValue())
      .Case<llvm::StringInit>([&](auto *init) {
        StringRef value = init->getValue();
        return value.empty() ? Optional<StringRef>() : value;
      })
      .Default([](auto *) { return std::nullopt; });
}

// Return the C++ class name for this type (which may just be ::mlir::Type).
std::string TypeConstraint::getCPPClassName() const {
  StringRef className = def->getValueAsString("cppClassName");

  // If the class name is already namespace resolved, use it.
  if (className.contains("::"))
    return className.str();

  // Otherwise, check to see if there is a namespace from a dialect to prepend.
  if (const llvm::RecordVal *value = def->getValue("dialect")) {
    Dialect dialect(cast<const llvm::DefInit>(value->getValue())->getDef());
    return (dialect.getCppNamespace() + "::" + className).str();
  }
  return className.str();
}

Type::Type(const llvm::Record *record) : TypeConstraint(record) {}

Dialect Type::getDialect() const {
  return Dialect(def->getValueAsDef("dialect"));
}
