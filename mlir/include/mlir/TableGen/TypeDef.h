//===-- TypeDef.h - Record wrapper for type definitions ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TypeDef wrapper to simplify using TableGen Record defining a MLIR type.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_TYPEDEF_H
#define MLIR_TABLEGEN_TYPEDEF_H

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Dialect.h"

namespace llvm {
class Record;
class DagInit;
class SMLoc;
} // namespace llvm

namespace mlir {
namespace tblgen {

class TypeParameter;

/// Wrapper class that contains a TableGen TypeDef's record and provides helper
/// methods for accessing them.
class TypeDef {
public:
  explicit TypeDef(const llvm::Record *def) : def(def) {}

  // Get the dialect for which this type belongs.
  Dialect getDialect() const;

  // Returns the name of this TypeDef record.
  StringRef getName() const;

  // Query functions for the documentation of the operator.
  bool hasDescription() const;
  StringRef getDescription() const;
  bool hasSummary() const;
  StringRef getSummary() const;

  // Returns the name of the C++ class to generate.
  StringRef getCppClassName() const;

  // Returns the name of the storage class for this type.
  StringRef getStorageClassName() const;

  // Returns the C++ namespace for this types storage class.
  StringRef getStorageNamespace() const;

  // Returns true if we should generate the storage class.
  bool genStorageClass() const;

  // Indicates whether or not to generate the storage class constructor.
  bool hasStorageCustomConstructor() const;

  // Fill a list with this types parameters. See TypeDef in OpBase.td for
  // documentation of parameter usage.
  void getParameters(SmallVectorImpl<TypeParameter> &) const;
  // Return the number of type parameters
  unsigned getNumParameters() const;

  // Return the keyword/mnemonic to use in the printer/parser methods if we are
  // supposed to auto-generate them.
  Optional<StringRef> getMnemonic() const;

  // Returns the code to use as the types printer method. If not specified,
  // return a non-value. Otherwise, return the contents of that code block.
  Optional<StringRef> getPrinterCode() const;

  // Returns the code to use as the types parser method. If not specified,
  // return a non-value. Otherwise, return the contents of that code block.
  Optional<StringRef> getParserCode() const;

  // Returns true if the accessors based on the types parameters should be
  // generated.
  bool genAccessors() const;

  // Return true if we need to generate the verifyConstructionInvariants
  // declaration and getChecked method.
  bool genVerifyInvariantsDecl() const;

  // Returns the dialects extra class declaration code.
  Optional<StringRef> getExtraDecls() const;

  // Get the code location (for error printing).
  ArrayRef<llvm::SMLoc> getLoc() const;

  // Returns whether two TypeDefs are equal by checking the equality of the
  // underlying record.
  bool operator==(const TypeDef &other) const;

  // Compares two TypeDefs by comparing the names of the dialects.
  bool operator<(const TypeDef &other) const;

  // Returns whether the TypeDef is defined.
  operator bool() const { return def != nullptr; }

private:
  const llvm::Record *def;
};

// A wrapper class for tblgen TypeParameter, arrays of which belong to TypeDefs
// to parameterize them.
class TypeParameter {
public:
  explicit TypeParameter(const llvm::DagInit *def, unsigned num)
      : def(def), num(num) {}

  // Get the parameter name.
  StringRef getName() const;
  // If specified, get the custom allocator code for this parameter.
  llvm::Optional<StringRef> getAllocator() const;
  // Get the C++ type of this parameter.
  StringRef getCppType() const;
  // Get a description of this parameter for documentation purposes.
  llvm::Optional<StringRef> getDescription() const;
  // Get the assembly syntax documentation.
  StringRef getSyntax() const;

private:
  const llvm::DagInit *def;
  const unsigned num;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_TYPEDEF_H
