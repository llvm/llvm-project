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

#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

using llvm::CodeInit;
using llvm::DefInit;
using llvm::Init;
using llvm::Record;
using llvm::StringInit;

// Returns the initializer's value as string if the given TableGen initializer
// is a code or string initializer. Returns the empty StringRef otherwise.
static StringRef getValueAsString(const Init *init) {
  if (const auto *code = dyn_cast<CodeInit>(init))
    return code->getValue().trim();
  if (const auto *str = dyn_cast<StringInit>(init))
    return str->getValue().trim();
  return {};
}

AttrConstraint::AttrConstraint(const Record *record)
    : Constraint(Constraint::CK_Attr, record) {
  assert(isSubClassOf("AttrConstraint") &&
         "must be subclass of TableGen 'AttrConstraint' class");
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
    return "Attribute";
  return type;
}

StringRef Attribute::getReturnType() const {
  const auto *init = def->getValueInit("returnType");
  return getValueAsString(init);
}

// Return the type constraint corresponding to the type of this attribute, or
// None if this is not a TypedAttr.
llvm::Optional<Type> Attribute::getValueType() const {
  if (auto *defInit = dyn_cast<llvm::DefInit>(def->getValueInit("valueType")))
    return Type(defInit->getDef());
  return llvm::None;
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
  if (const auto *defInit =
          llvm::dyn_cast<llvm::DefInit>(def->getValueInit("baseAttr"))) {
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
    if (DefInit *init = dyn_cast<DefInit>(record->getValue()))
      return Dialect(init->getDef());
  }
  return Dialect(nullptr);
}

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

EnumAttrCase::EnumAttrCase(const llvm::Record *record) : Attribute(record) {
  assert(isSubClassOf("EnumAttrCaseInfo") &&
         "must be subclass of TableGen 'EnumAttrInfo' class");
}

EnumAttrCase::EnumAttrCase(const llvm::DefInit *init)
    : EnumAttrCase(init->getDef()) {}

bool EnumAttrCase::isStrCase() const { return isSubClassOf("StrEnumAttrCase"); }

StringRef EnumAttrCase::getSymbol() const {
  return def->getValueAsString("symbol");
}

StringRef EnumAttrCase::getStr() const { return def->getValueAsString("str"); }

int64_t EnumAttrCase::getValue() const { return def->getValueAsInt("value"); }

const llvm::Record &EnumAttrCase::getDef() const { return *def; }

EnumAttr::EnumAttr(const llvm::Record *record) : Attribute(record) {
  assert(isSubClassOf("EnumAttrInfo") &&
         "must be subclass of TableGen 'EnumAttr' class");
}

EnumAttr::EnumAttr(const llvm::Record &record) : Attribute(&record) {}

EnumAttr::EnumAttr(const llvm::DefInit *init) : EnumAttr(init->getDef()) {}

bool EnumAttr::classof(const Attribute *attr) {
  return attr->isSubClassOf("EnumAttrInfo");
}

bool EnumAttr::isBitEnum() const { return isSubClassOf("BitEnumAttr"); }

StringRef EnumAttr::getEnumClassName() const {
  return def->getValueAsString("className");
}

StringRef EnumAttr::getCppNamespace() const {
  return def->getValueAsString("cppNamespace");
}

StringRef EnumAttr::getUnderlyingType() const {
  return def->getValueAsString("underlyingType");
}

StringRef EnumAttr::getUnderlyingToSymbolFnName() const {
  return def->getValueAsString("underlyingToSymbolFnName");
}

StringRef EnumAttr::getStringToSymbolFnName() const {
  return def->getValueAsString("stringToSymbolFnName");
}

StringRef EnumAttr::getSymbolToStringFnName() const {
  return def->getValueAsString("symbolToStringFnName");
}

StringRef EnumAttr::getSymbolToStringFnRetType() const {
  return def->getValueAsString("symbolToStringFnRetType");
}

StringRef EnumAttr::getMaxEnumValFnName() const {
  return def->getValueAsString("maxEnumValFnName");
}

std::vector<EnumAttrCase> EnumAttr::getAllCases() const {
  const auto *inits = def->getValueAsListInit("enumerants");

  std::vector<EnumAttrCase> cases;
  cases.reserve(inits->size());

  for (const llvm::Init *init : *inits) {
    cases.push_back(EnumAttrCase(cast<llvm::DefInit>(init)));
  }

  return cases;
}

StructFieldAttr::StructFieldAttr(const llvm::Record *record) : def(record) {
  assert(def->isSubClassOf("StructFieldAttr") &&
         "must be subclass of TableGen 'StructFieldAttr' class");
}

StructFieldAttr::StructFieldAttr(const llvm::Record &record)
    : StructFieldAttr(&record) {}

StructFieldAttr::StructFieldAttr(const llvm::DefInit *init)
    : StructFieldAttr(init->getDef()) {}

StringRef StructFieldAttr::getName() const {
  return def->getValueAsString("name");
}

Attribute StructFieldAttr::getType() const {
  auto init = def->getValueInit("type");
  return Attribute(cast<llvm::DefInit>(init));
}

StructAttr::StructAttr(const llvm::Record *record) : Attribute(record) {
  assert(isSubClassOf("StructAttr") &&
         "must be subclass of TableGen 'StructAttr' class");
}

StructAttr::StructAttr(const llvm::DefInit *init)
    : StructAttr(init->getDef()) {}

StringRef StructAttr::getStructClassName() const {
  return def->getValueAsString("className");
}

StringRef StructAttr::getCppNamespace() const {
  Dialect dialect(def->getValueAsDef("dialect"));
  return dialect.getCppNamespace();
}

std::vector<StructFieldAttr> StructAttr::getAllFields() const {
  std::vector<StructFieldAttr> attributes;

  const auto *inits = def->getValueAsListInit("fields");
  attributes.reserve(inits->size());

  for (const llvm::Init *init : *inits) {
    attributes.emplace_back(cast<llvm::DefInit>(init));
  }

  return attributes;
}

const char * ::mlir::tblgen::inferTypeOpInterface = "InferTypeOpInterface";
