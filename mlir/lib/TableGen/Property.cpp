//===- Property.cpp - Property wrapper class ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Property wrapper to simplify using TableGen Record defining a MLIR
// Property.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Property.h"
#include "mlir/TableGen/Format.h"
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

Property::Property(const Record *record) : def(record) {
  assert((record->isSubClassOf("Property") || record->isSubClassOf("Attr")) &&
         "must be subclass of TableGen 'Property' class");
}

Property::Property(const DefInit *init) : Property(init->getDef()) {}

StringRef Property::getStorageType() const {
  const auto *init = def->getValueInit("storageType");
  auto type = getValueAsString(init);
  if (type.empty())
    return "Property";
  return type;
}

StringRef Property::getInterfaceType() const {
  const auto *init = def->getValueInit("interfaceType");
  return getValueAsString(init);
}

StringRef Property::getConvertFromStorageCall() const {
  const auto *init = def->getValueInit("convertFromStorage");
  return getValueAsString(init);
}

StringRef Property::getAssignToStorageCall() const {
  const auto *init = def->getValueInit("assignToStorage");
  return getValueAsString(init);
}

StringRef Property::getConvertToAttributeCall() const {
  const auto *init = def->getValueInit("convertToAttribute");
  return getValueAsString(init);
}

StringRef Property::getConvertFromAttributeCall() const {
  const auto *init = def->getValueInit("convertFromAttribute");
  return getValueAsString(init);
}

StringRef Property::getHashPropertyCall() const {
  return getValueAsString(def->getValueInit("hashProperty"));
}

bool Property::hasDefaultValue() const { return !getDefaultValue().empty(); }

StringRef Property::getDefaultValue() const {
  const auto *init = def->getValueInit("defaultValue");
  return getValueAsString(init);
}

const llvm::Record &Property::getDef() const { return *def; }
