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

namespace mlir {
namespace tblgen {

StringRef tblgen::Dialect::getName() const {
  return def->getValueAsString("name");
}

StringRef tblgen::Dialect::getCppNamespace() const {
  return def->getValueAsString("cppNamespace");
}

std::string tblgen::Dialect::getCppClassName() const {
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

StringRef tblgen::Dialect::getSummary() const {
  return getAsStringOrEmpty(*def, "summary");
}

StringRef tblgen::Dialect::getDescription() const {
  return getAsStringOrEmpty(*def, "description");
}

llvm::Optional<StringRef> tblgen::Dialect::getExtraClassDeclaration() const {
  auto value = def->getValueAsString("extraClassDeclaration");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

bool tblgen::Dialect::hasConstantMaterializer() const {
  return def->getValueAsBit("hasConstantMaterializer");
}

bool tblgen::Dialect::hasOperationAttrVerify() const {
  return def->getValueAsBit("hasOperationAttrVerify");
}

bool tblgen::Dialect::hasRegionArgAttrVerify() const {
  return def->getValueAsBit("hasRegionArgAttrVerify");
}

bool tblgen::Dialect::hasRegionResultAttrVerify() const {
  return def->getValueAsBit("hasRegionResultAttrVerify");
}

bool Dialect::operator==(const Dialect &other) const {
  return def == other.def;
}

bool Dialect::operator<(const Dialect &other) const {
  return getName() < other.getName();
}

} // end namespace tblgen
} // end namespace mlir
