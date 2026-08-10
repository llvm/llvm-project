//===- EnumInfo.cpp - EnumInfo wrapper class ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/EnumInfo.h"
#include "mlir/TableGen/Attribute.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

using llvm::DefInit;
using llvm::Init;
using llvm::Record;

EnumCase::EnumCase(const Record *record) : def(record) {
  assert(def->isSubClassOf("EnumCase") &&
         "must be subclass of TableGen 'EnumCase' class");
}

EnumCase::EnumCase(const DefInit *init) : EnumCase(init->getDef()) {}

StringRef EnumCase::getSymbol() const {
  return def->getValueAsString("symbol");
}

StringRef EnumCase::getStr() const { return def->getValueAsString("str"); }

int64_t EnumCase::getValue() const { return def->getValueAsInt("value"); }

const Record &EnumCase::getDef() const { return *def; }

EnumInfo::EnumInfo(const Record *record) : def(record) {
  assert(isSubClassOf("EnumInfo") &&
         "must be subclass of TableGen 'EnumInfo' class");
}

EnumInfo::EnumInfo(const Record &record) : EnumInfo(&record) {}

EnumInfo::EnumInfo(const DefInit *init) : EnumInfo(init->getDef()) {}

bool EnumInfo::isSubClassOf(StringRef className) const {
  return def->isSubClassOf(className);
}

bool EnumInfo::isEnumAttr() const { return isSubClassOf("EnumAttrInfo"); }

std::optional<Attribute> EnumInfo::asEnumAttr() const {
  if (isEnumAttr())
    return Attribute(def);
  return std::nullopt;
}

bool EnumInfo::isBitEnum() const { return isSubClassOf("BitEnumBase"); }

StringRef EnumInfo::getEnumClassName() const {
  return def->getValueAsString("className");
}

StringRef EnumInfo::getSummary() const {
  return def->getValueAsString("summary");
}

StringRef EnumInfo::getDescription() const {
  return def->getValueAsString("description");
}

StringRef EnumInfo::getCppNamespace() const {
  return def->getValueAsString("cppNamespace");
}

int64_t EnumInfo::getBitwidth() const { return def->getValueAsInt("bitwidth"); }

StringRef EnumInfo::getUnderlyingType() const {
  return def->getValueAsString("underlyingType");
}

StringRef EnumInfo::getUnderlyingToSymbolFnName() const {
  return def->getValueAsString("underlyingToSymbolFnName");
}

StringRef EnumInfo::getStringToSymbolFnName() const {
  return def->getValueAsString("stringToSymbolFnName");
}

StringRef EnumInfo::getSymbolToStringFnName() const {
  return def->getValueAsString("symbolToStringFnName");
}

StringRef EnumInfo::getSymbolToStringFnRetType() const {
  return def->getValueAsString("symbolToStringFnRetType");
}

StringRef EnumInfo::getMaxEnumValFnName() const {
  return def->getValueAsString("maxEnumValFnName");
}

std::vector<EnumCase> EnumInfo::getAllCases() const {
  const auto *inits = def->getValueAsListInit("enumerants");

  std::vector<EnumCase> cases;
  cases.reserve(inits->size());

  for (const Init *init : *inits) {
    cases.emplace_back(cast<DefInit>(init));
  }

  return cases;
}

bool EnumInfo::genSpecializedAttr() const {
  return isSubClassOf("EnumAttrInfo") &&
         def->getValueAsBit("genSpecializedAttr");
}

const Record *EnumInfo::getBaseAttrClass() const {
  return def->getValueAsDef("baseAttrClass");
}

StringRef EnumInfo::getSpecializedAttrClassName() const {
  return def->getValueAsString("specializedAttrClassName");
}

bool EnumInfo::printBitEnumPrimaryGroups() const {
  return def->getValueAsBit("printBitEnumPrimaryGroups");
}

bool EnumInfo::printBitEnumQuoted() const {
  return def->getValueAsBit("printBitEnumQuoted");
}

const Record &EnumInfo::getDef() const { return *def; }
