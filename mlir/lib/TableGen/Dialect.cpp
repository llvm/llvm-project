//===- Dialect.cpp - Dialect wrapper class --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect wrapper to simplify using TableGen Record defining a MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Dialect.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
Dialect::Dialect(const llvm::Record *def) : def(def) {
  for (StringRef dialect : def->getValueAsListOfStrings("dependentDialects"))
    dependentDialects.push_back(dialect);
}

StringRef Dialect::getName() const { return def->getValueAsString("name"); }

StringRef Dialect::getCppNamespace() const {
  return def->getValueAsString("cppNamespace");
}

std::string Dialect::getCppClassName() const {
  // Simply use the name and remove any '_' tokens.
  std::string cppName = def->getName().str();
  llvm::erase_if(cppName, [](char c) { return c == '_'; });
  return cppName;
}

static StringRef getAsStringOrEmpty(const llvm::Record &record,
                                    StringRef fieldName) {
  if (auto valueInit = record.getValueInit(fieldName)) {
    if (llvm::isa<llvm::CodeInit, llvm::StringInit>(valueInit))
      return record.getValueAsString(fieldName);
  }
  return "";
}

StringRef Dialect::getSummary() const {
  return getAsStringOrEmpty(*def, "summary");
}

StringRef Dialect::getDescription() const {
  return getAsStringOrEmpty(*def, "description");
}

ArrayRef<StringRef> Dialect::getDependentDialects() const {
  return dependentDialects;
}

llvm::Optional<StringRef> Dialect::getExtraClassDeclaration() const {
  auto value = def->getValueAsString("extraClassDeclaration");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

bool Dialect::hasConstantMaterializer() const {
  return def->getValueAsBit("hasConstantMaterializer");
}

bool Dialect::hasOperationAttrVerify() const {
  return def->getValueAsBit("hasOperationAttrVerify");
}

bool Dialect::hasRegionArgAttrVerify() const {
  return def->getValueAsBit("hasRegionArgAttrVerify");
}

bool Dialect::hasRegionResultAttrVerify() const {
  return def->getValueAsBit("hasRegionResultAttrVerify");
}

bool Dialect::operator==(const Dialect &other) const {
  return def == other.def;
}

bool Dialect::operator<(const Dialect &other) const {
  return getName() < other.getName();
}
