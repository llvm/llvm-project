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
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
Dialect::Dialect(const llvm::Record *def) : def(def) {
  if (def == nullptr)
    return;
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
  llvm::erase_value(cppName, '_');
  return cppName;
}

static StringRef getAsStringOrEmpty(const llvm::Record &record,
                                    StringRef fieldName) {
  if (auto *valueInit = record.getValueInit(fieldName)) {
    if (llvm::isa<llvm::StringInit>(valueInit))
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

std::optional<StringRef> Dialect::getExtraClassDeclaration() const {
  auto value = def->getValueAsString("extraClassDeclaration");
  return value.empty() ? std::optional<StringRef>() : value;
}

bool Dialect::hasCanonicalizer() const {
  return def->getValueAsBit("hasCanonicalizer");
}

bool Dialect::hasConstantMaterializer() const {
  return def->getValueAsBit("hasConstantMaterializer");
}

bool Dialect::hasNonDefaultDestructor() const {
  return def->getValueAsBit("hasNonDefaultDestructor");
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

bool Dialect::hasOperationInterfaceFallback() const {
  return def->getValueAsBit("hasOperationInterfaceFallback");
}

bool Dialect::useDefaultAttributePrinterParser() const {
  return def->getValueAsBit("useDefaultAttributePrinterParser");
}

bool Dialect::useDefaultTypePrinterParser() const {
  return def->getValueAsBit("useDefaultTypePrinterParser");
}

bool Dialect::isExtensible() const {
  return def->getValueAsBit("isExtensible");
}

Dialect::FolderAPI Dialect::getFolderAPI() const {
  int64_t value = def->getValueAsInt("useFoldAPI");
  if (value < static_cast<int64_t>(FolderAPI::RawAttributes) ||
      value > static_cast<int64_t>(FolderAPI::FolderAdaptor))
    llvm::PrintFatalError(def->getLoc(),
                          "Invalid value for dialect field `useFoldAPI`");

  return static_cast<FolderAPI>(value);
}

bool Dialect::operator==(const Dialect &other) const {
  return def == other.def;
}

bool Dialect::operator<(const Dialect &other) const {
  return getName() < other.getName();
}
