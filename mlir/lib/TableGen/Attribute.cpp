//===- Attribute.cpp - Attribute wrapper class ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attribute wrapper to simplify using TableGen Record defining a MLIR
// Attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Operator.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

using llvm::DefInit;
using llvm::Init;
using llvm::Record;
using llvm::StringInit;

// Returns the initializer's value as string if the given TableGen initializer
// is a code or string initializer. Returns the empty StringRef otherwise.
static StringRef getValueAsString(const Init *init) {
  if (const auto *str = dyn_cast<StringInit>(init))
    return str->getValue().trim();
  return {};
}

bool AttrConstraint::isSubClassOf(StringRef className) const {
  return def->isSubClassOf(className);
}

Attribute::Attribute(const Record *record) : AttrConstraint(record) {
  assert(record->isSubClassOf("Attr") &&
         "must be subclass of TableGen 'Attr' class");
}

Attribute::Attribute(const DefInit *init) : Attribute(init->getDef()) {}

bool Attribute::isDerivedAttr() const { return isSubClassOf("DerivedAttr"); }

bool Attribute::isTypeAttr() const { return isSubClassOf("TypeAttrBase"); }

bool Attribute::isSymbolRefAttr() const {
  StringRef defName = def->getName();
  if (defName == "SymbolRefAttr" || defName == "FlatSymbolRefAttr")
    return true;
  return isSubClassOf("SymbolRefAttr") || isSubClassOf("FlatSymbolRefAttr");
}

bool Attribute::isEnumAttr() const { return isSubClassOf("EnumAttrInfo"); }

StringRef Attribute::getStorageType() const {
  const auto *init = def->getValueInit("storageType");
  auto type = getValueAsString(init);
  if (type.empty())
    return "::mlir::Attribute";
  return type;
}

StringRef Attribute::getReturnType() const {
  const auto *init = def->getValueInit("returnType");
  return getValueAsString(init);
}

// Return the type constraint corresponding to the type of this attribute, or
// std::nullopt if this is not a TypedAttr.
std::optional<Type> Attribute::getValueType() const {
  if (const auto *defInit = dyn_cast<DefInit>(def->getValueInit("valueType")))
    return Type(defInit->getDef());
  return std::nullopt;
}

StringRef Attribute::getConvertFromStorageCall() const {
  const auto *init = def->getValueInit("convertFromStorage");
  return getValueAsString(init);
}

bool Attribute::isConstBuildable() const {
  const auto *init = def->getValueInit("constBuilderCall");
  return !getValueAsString(init).empty();
}

StringRef Attribute::getConstBuilderTemplate() const {
  const auto *init = def->getValueInit("constBuilderCall");
  return getValueAsString(init);
}

Attribute Attribute::getBaseAttr() const {
  if (const auto *defInit = dyn_cast<DefInit>(def->getValueInit("baseAttr"))) {
    return Attribute(defInit).getBaseAttr();
  }
  return *this;
}

bool Attribute::hasDefaultValue() const {
  const auto *init = def->getValueInit("defaultValue");
  return !getValueAsString(init).empty();
}

StringRef Attribute::getDefaultValue() const {
  const auto *init = def->getValueInit("defaultValue");
  return getValueAsString(init);
}

bool Attribute::isOptional() const { return def->getValueAsBit("isOptional"); }

StringRef Attribute::getAttrDefName() const {
  if (def->isAnonymous()) {
    return getBaseAttr().def->getName();
  }
  return def->getName();
}

StringRef Attribute::getDerivedCodeBody() const {
  assert(isDerivedAttr() && "only derived attribute has 'body' field");
  return def->getValueAsString("body");
}

Dialect Attribute::getDialect() const {
  const llvm::RecordVal *record = def->getValue("dialect");
  if (record && record->getValue()) {
    if (const DefInit *init = dyn_cast<DefInit>(record->getValue()))
      return Dialect(init->getDef());
  }
  return Dialect(nullptr);
}

const Record &Attribute::getDef() const { return *def; }

ConstantAttr::ConstantAttr(const DefInit *init) : def(init->getDef()) {
  assert(def->isSubClassOf("ConstantAttr") &&
         "must be subclass of TableGen 'ConstantAttr' class");
}

Attribute ConstantAttr::getAttribute() const {
  return Attribute(def->getValueAsDef("attr"));
}

StringRef ConstantAttr::getConstantValue() const {
  return def->getValueAsString("value");
}

const char * ::mlir::tblgen::inferTypeOpInterface = "InferTypeOpInterface";
