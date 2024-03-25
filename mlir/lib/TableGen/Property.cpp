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

Property::Property(const Record *def)
    : Property(getValueAsString(def->getValueInit("storageType")),
               getValueAsString(def->getValueInit("interfaceType")),
               getValueAsString(def->getValueInit("convertFromStorage")),
               getValueAsString(def->getValueInit("assignToStorage")),
               getValueAsString(def->getValueInit("convertToAttribute")),
               getValueAsString(def->getValueInit("convertFromAttribute")),
               getValueAsString(def->getValueInit("readFromMlirBytecode")),
               getValueAsString(def->getValueInit("writeToMlirBytecode")),
               getValueAsString(def->getValueInit("hashProperty")),
               getValueAsString(def->getValueInit("defaultValue"))) {
  this->def = def;
  assert((def->isSubClassOf("Property") || def->isSubClassOf("Attr")) &&
         "must be subclass of TableGen 'Property' class");
}

Property::Property(const DefInit *init) : Property(init->getDef()) {}

Property::Property(StringRef storageType, StringRef interfaceType,
                   StringRef convertFromStorageCall,
                   StringRef assignToStorageCall,
                   StringRef convertToAttributeCall,
                   StringRef convertFromAttributeCall,
                   StringRef readFromMlirBytecodeCall,
                   StringRef writeToMlirBytecodeCall,
                   StringRef hashPropertyCall, StringRef defaultValue)
    : storageType(storageType), interfaceType(interfaceType),
      convertFromStorageCall(convertFromStorageCall),
      assignToStorageCall(assignToStorageCall),
      convertToAttributeCall(convertToAttributeCall),
      convertFromAttributeCall(convertFromAttributeCall),
      readFromMlirBytecodeCall(readFromMlirBytecodeCall),
      writeToMlirBytecodeCall(writeToMlirBytecodeCall),
      hashPropertyCall(hashPropertyCall), defaultValue(defaultValue) {
  if (storageType.empty())
    storageType = "Property";
}
