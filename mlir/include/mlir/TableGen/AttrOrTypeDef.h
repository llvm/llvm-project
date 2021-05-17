//===-- AttrOrTypeDef.h - Wrapper for attr and type definitions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AttrOrTypeDef, AttrDef, and TypeDef wrappers to simplify using TableGen
// Record defining a MLIR attributes and types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_ATTRORTYPEDEF_H
#define MLIR_TABLEGEN_ATTRORTYPEDEF_H

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Builder.h"
#include "mlir/TableGen/Trait.h"

namespace llvm {
class DagInit;
class Record;
class SMLoc;
} // namespace llvm

namespace mlir {
namespace tblgen {
class Dialect;
class AttrOrTypeParameter;

//===----------------------------------------------------------------------===//
// AttrOrTypeBuilder
//===----------------------------------------------------------------------===//

/// Wrapper class that represents a Tablegen AttrOrTypeBuilder.
class AttrOrTypeBuilder : public Builder {
public:
  using Builder::Builder;

  /// Returns true if this builder is able to infer the MLIRContext parameter.
  bool hasInferredContextParameter() const;
};

//===----------------------------------------------------------------------===//
// AttrOrTypeDef
//===----------------------------------------------------------------------===//

/// Wrapper class that contains a TableGen AttrOrTypeDef's record and provides
/// helper methods for accessing them.
class AttrOrTypeDef {
public:
  explicit AttrOrTypeDef(const llvm::Record *def);

  // Get the dialect for which this def belongs.
  Dialect getDialect() const;

  // Returns the name of this AttrOrTypeDef record.
  StringRef getName() const;

  // Query functions for the documentation of the def.
  bool hasDescription() const;
  StringRef getDescription() const;
  bool hasSummary() const;
  StringRef getSummary() const;

  // Returns the name of the C++ class to generate.
  StringRef getCppClassName() const;

  // Returns the name of the C++ base class to use when generating this def.
  StringRef getCppBaseClassName() const;

  // Returns the name of the storage class for this def.
  StringRef getStorageClassName() const;

  // Returns the C++ namespace for this def's storage class.
  StringRef getStorageNamespace() const;

  // Returns true if we should generate the storage class.
  bool genStorageClass() const;

  // Indicates whether or not to generate the storage class constructor.
  bool hasStorageCustomConstructor() const;

  // Fill a list with this def's parameters. See AttrOrTypeDef in OpBase.td for
  // documentation of parameter usage.
  void getParameters(SmallVectorImpl<AttrOrTypeParameter> &) const;

  // Return the number of parameters
  unsigned getNumParameters() const;

  // Return the keyword/mnemonic to use in the printer/parser methods if we are
  // supposed to auto-generate them.
  Optional<StringRef> getMnemonic() const;

  // Returns the code to use as the types printer method. If not specified,
  // return a non-value. Otherwise, return the contents of that code block.
  Optional<StringRef> getPrinterCode() const;

  // Returns the code to use as the parser method. If not specified, returns
  // None. Otherwise, returns the contents of that code block.
  Optional<StringRef> getParserCode() const;

  // Returns true if the accessors based on the parameters should be generated.
  bool genAccessors() const;

  // Return true if we need to generate the verify declaration and getChecked
  // method.
  bool genVerifyDecl() const;

  // Returns the def's extra class declaration code.
  Optional<StringRef> getExtraDecls() const;

  // Get the code location (for error printing).
  ArrayRef<llvm::SMLoc> getLoc() const;

  // Returns true if the default get/getChecked methods should be skipped during
  // generation.
  bool skipDefaultBuilders() const;

  // Returns the builders of this def.
  ArrayRef<AttrOrTypeBuilder> getBuilders() const { return builders; }

  // Returns the traits of this def.
  ArrayRef<Trait> getTraits() const { return traits; }

  // Returns whether two AttrOrTypeDefs are equal by checking the equality of
  // the underlying record.
  bool operator==(const AttrOrTypeDef &other) const;

  // Compares two AttrOrTypeDefs by comparing the names of the dialects.
  bool operator<(const AttrOrTypeDef &other) const;

  // Returns whether the AttrOrTypeDef is defined.
  operator bool() const { return def != nullptr; }

  // Return the underlying def.
  const llvm::Record *getDef() const { return def; }

protected:
  const llvm::Record *def;

  // The builders of this definition.
  SmallVector<AttrOrTypeBuilder> builders;

  // The traits of this definition.
  SmallVector<Trait> traits;
};

//===----------------------------------------------------------------------===//
// AttrDef
//===----------------------------------------------------------------------===//

/// This class represents a wrapper around a tablegen AttrDef record.
class AttrDef : public AttrOrTypeDef {
public:
  using AttrOrTypeDef::AttrOrTypeDef;

  // Returns the attributes value type builder code block, or None if it doesn't
  // have one.
  Optional<StringRef> getTypeBuilder() const;

  static bool classof(const AttrOrTypeDef *def);
};

//===----------------------------------------------------------------------===//
// TypeDef
//===----------------------------------------------------------------------===//

/// This class represents a wrapper around a tablegen TypeDef record.
class TypeDef : public AttrOrTypeDef {
public:
  using AttrOrTypeDef::AttrOrTypeDef;
};

//===----------------------------------------------------------------------===//
// AttrOrTypeParameter
//===----------------------------------------------------------------------===//

// A wrapper class for tblgen AttrOrTypeParameter, arrays of which belong to
// AttrOrTypeDefs to parameterize them.
class AttrOrTypeParameter {
public:
  explicit AttrOrTypeParameter(const llvm::DagInit *def, unsigned index)
      : def(def), index(index) {}

  // Get the parameter name.
  StringRef getName() const;

  // If specified, get the custom allocator code for this parameter.
  Optional<StringRef> getAllocator() const;

  // If specified, get the custom comparator code for this parameter.
  Optional<StringRef> getComparator() const;

  // Get the C++ type of this parameter.
  StringRef getCppType() const;

  // Get a description of this parameter for documentation purposes.
  Optional<StringRef> getSummary() const;

  // Get the assembly syntax documentation.
  StringRef getSyntax() const;

  // Return the underlying def of this parameter.
  const llvm::Init *getDef() const;

private:
  /// The underlying tablegen parameter list this parameter is a part of.
  const llvm::DagInit *def;
  /// The index of the parameter within the parameter list (`def`).
  unsigned index;
};

//===----------------------------------------------------------------------===//
// AttributeSelfTypeParameter
//===----------------------------------------------------------------------===//

// A wrapper class for the AttributeSelfTypeParameter tblgen class. This
// represents a parameter of mlir::Type that is the value type of an AttrDef.
class AttributeSelfTypeParameter : public AttrOrTypeParameter {
public:
  static bool classof(const AttrOrTypeParameter *param);
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_ATTRORTYPEDEF_H
