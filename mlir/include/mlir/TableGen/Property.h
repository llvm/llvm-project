//===- Property.h - Property wrapper class --------------------*- C++ -*-===//
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

#ifndef MLIR_TABLEGEN_PROPERTY_H_
#define MLIR_TABLEGEN_PROPERTY_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Constraint.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {
class Dialect;
class Type;

// Wrapper class providing helper methods for accessing MLIR Property defined
// in TableGen. This class should closely reflect what is defined as class
// `Property` in TableGen.
class Property {
public:
  explicit Property(const llvm::Record *record);
  explicit Property(const llvm::DefInit *init);
  Property(StringRef storageType, StringRef interfaceType,
           StringRef convertFromStorageCall, StringRef assignToStorageCall,
           StringRef convertToAttributeCall, StringRef convertFromAttributeCall,
           StringRef readFromMlirBytecodeCall,
           StringRef writeToMlirBytecodeCall, StringRef hashPropertyCall,
           StringRef defaultValue);

  // Returns the storage type.
  StringRef getStorageType() const { return storageType; }

  // Returns the interface type for this property.
  StringRef getInterfaceType() const { return interfaceType; }

  // Returns the template getter method call which reads this property's
  // storage and returns the value as of the desired return type.
  StringRef getConvertFromStorageCall() const { return convertFromStorageCall; }

  // Returns the template setter method call which reads this property's
  // in the provided interface type and assign it to the storage.
  StringRef getAssignToStorageCall() const { return assignToStorageCall; }

  // Returns the conversion method call which reads this property's
  // in the storage type and builds an attribute.
  StringRef getConvertToAttributeCall() const { return convertToAttributeCall; }

  // Returns the setter method call which reads this property's
  // in the provided interface type and assign it to the storage.
  StringRef getConvertFromAttributeCall() const {
    return convertFromAttributeCall;
  }

  // Returns the method call which reads this property from
  // bytecode and assign it to the storage.
  StringRef getReadFromMlirBytecodeCall() const {
    return readFromMlirBytecodeCall;
  }

  // Returns the method call which write this property's
  // to the the bytecode.
  StringRef getWriteToMlirBytecodeCall() const {
    return writeToMlirBytecodeCall;
  }

  // Returns the code to compute the hash for this property.
  StringRef getHashPropertyCall() const { return hashPropertyCall; }

  // Returns whether this Property has a default value.
  bool hasDefaultValue() const { return !defaultValue.empty(); }

  // Returns the default value for this Property.
  StringRef getDefaultValue() const { return defaultValue; }

  // Returns the TableGen definition this Property was constructed from.
  const llvm::Record &getDef() const { return *def; }

private:
  // The TableGen definition of this constraint.
  const llvm::Record *def;

  // Elements describing a Property, in general fetched from the record.
  StringRef storageType;
  StringRef interfaceType;
  StringRef convertFromStorageCall;
  StringRef assignToStorageCall;
  StringRef convertToAttributeCall;
  StringRef convertFromAttributeCall;
  StringRef readFromMlirBytecodeCall;
  StringRef writeToMlirBytecodeCall;
  StringRef hashPropertyCall;
  StringRef defaultValue;
};

// A struct wrapping an op property and its name together
struct NamedProperty {
  llvm::StringRef name;
  Property prop;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PROPERTY_H_
