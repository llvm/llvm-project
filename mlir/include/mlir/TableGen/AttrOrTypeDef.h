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

//===----------------------------------------------------------------------===//
// AttrOrTypeBuilder
//===----------------------------------------------------------------------===//

/// Wrapper class that represents a Tablegen AttrOrTypeBuilder.
class AttrOrTypeBuilder : public Builder {
public:
  using Builder::Builder;

  /// Returns an optional builder return type.
  Optional<StringRef> getReturnType() const;

  /// Returns true if this builder is able to infer the MLIRContext parameter.
  bool hasInferredContextParameter() const;
};

//===----------------------------------------------------------------------===//
// AttrOrTypeParameter
//===----------------------------------------------------------------------===//

/// A wrapper class for tblgen AttrOrTypeParameter, arrays of which belong to
/// AttrOrTypeDefs to parameterize them.
class AttrOrTypeParameter {
public:
  explicit AttrOrTypeParameter(const llvm::DagInit *def, unsigned index)
      : def(def), index(index) {}

  /// Returns true if the parameter is anonymous (has no name).
  bool isAnonymous() const;

  /// Get the parameter name.
  StringRef getName() const;

  /// Get the parameter accessor name.
  std::string getAccessorName() const;

  /// If specified, get the custom allocator code for this parameter.
  Optional<StringRef> getAllocator() const;

  /// If specified, get the custom comparator code for this parameter.
  StringRef getComparator() const;

  /// Get the C++ type of this parameter.
  StringRef getCppType() const;

  /// Get the C++ accessor type of this parameter.
  StringRef getCppAccessorType() const;

  /// Get the C++ storage type of this parameter.
  StringRef getCppStorageType() const;

  /// Get the C++ code to convert from the storage type to the parameter type.
  StringRef getConvertFromStorage() const;

  /// Get an optional C++ parameter parser.
  Optional<StringRef> getParser() const;

  /// Get an optional C++ parameter printer.
  Optional<StringRef> getPrinter() const;

  /// Get a description of this parameter for documentation purposes.
  Optional<StringRef> getSummary() const;

  /// Get the assembly syntax documentation.
  StringRef getSyntax() const;

  /// Returns true if the parameter is optional.
  bool isOptional() const;

  /// Get the default value of the parameter if it has one.
  Optional<StringRef> getDefaultValue() const;

  /// Return the underlying def of this parameter.
  llvm::Init *getDef() const;

  /// The parameter is pointer-comparable.
  bool operator==(const AttrOrTypeParameter &other) const {
    return def == other.def && index == other.index;
  }
  bool operator!=(const AttrOrTypeParameter &other) const {
    return !(*this == other);
  }

private:
  /// A parameter can be either a string or a def. Get a potentially null value
  /// from the def.
  template <typename InitT>
  auto getDefValue(StringRef name) const;

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

//===----------------------------------------------------------------------===//
// AttrOrTypeDef
//===----------------------------------------------------------------------===//

/// Wrapper class that contains a TableGen AttrOrTypeDef's record and provides
/// helper methods for accessing them.
class AttrOrTypeDef {
public:
  explicit AttrOrTypeDef(const llvm::Record *def);

  /// Get the dialect for which this def belongs.
  Dialect getDialect() const;

  /// Returns the name of this AttrOrTypeDef record.
  StringRef getName() const;

  /// Query functions for the documentation of the def.
  bool hasDescription() const;
  StringRef getDescription() const;
  bool hasSummary() const;
  StringRef getSummary() const;

  /// Returns the name of the C++ class to generate.
  StringRef getCppClassName() const;

  /// Returns the name of the C++ base class to use when generating this def.
  StringRef getCppBaseClassName() const;

  /// Returns the name of the storage class for this def.
  StringRef getStorageClassName() const;

  /// Returns the C++ namespace for this def's storage class.
  StringRef getStorageNamespace() const;

  /// Returns true if we should generate the storage class.
  bool genStorageClass() const;

  /// Indicates whether or not to generate the storage class constructor.
  bool hasStorageCustomConstructor() const;

  /// Get the parameters of this attribute or type.
  ArrayRef<AttrOrTypeParameter> getParameters() const { return parameters; }

  /// Return the number of parameters
  unsigned getNumParameters() const;

  /// Return the keyword/mnemonic to use in the printer/parser methods if we are
  /// supposed to auto-generate them.
  Optional<StringRef> getMnemonic() const;

  /// Returns if the attribute or type has a custom assembly format implemented
  /// in C++. Corresponds to the `hasCustomAssemblyFormat` field.
  bool hasCustomAssemblyFormat() const;

  /// Returns the custom assembly format, if one was specified.
  Optional<StringRef> getAssemblyFormat() const;

  /// Returns true if the accessors based on the parameters should be generated.
  bool genAccessors() const;

  /// Return true if we need to generate the verify declaration and getChecked
  /// method.
  bool genVerifyDecl() const;

  /// Returns the def's extra class declaration code.
  Optional<StringRef> getExtraDecls() const;

  /// Returns the def's extra class definition code.
  Optional<StringRef> getExtraDefs() const;

  /// Get the code location (for error printing).
  ArrayRef<SMLoc> getLoc() const;

  /// Returns true if the default get/getChecked methods should be skipped
  /// during generation.
  bool skipDefaultBuilders() const;

  /// Returns the builders of this def.
  ArrayRef<AttrOrTypeBuilder> getBuilders() const { return builders; }

  /// Returns the traits of this def.
  ArrayRef<Trait> getTraits() const { return traits; }

  /// Returns whether two AttrOrTypeDefs are equal by checking the equality of
  /// the underlying record.
  bool operator==(const AttrOrTypeDef &other) const;

  /// Compares two AttrOrTypeDefs by comparing the names of the dialects.
  bool operator<(const AttrOrTypeDef &other) const;

  /// Returns whether the AttrOrTypeDef is defined.
  operator bool() const { return def != nullptr; }

  /// Return the underlying def.
  const llvm::Record *getDef() const { return def; }

protected:
  const llvm::Record *def;

  /// The builders of this definition.
  SmallVector<AttrOrTypeBuilder> builders;

  /// The traits of this definition.
  SmallVector<Trait> traits;

  /// The parameters of this attribute or type.
  SmallVector<AttrOrTypeParameter> parameters;
};

//===----------------------------------------------------------------------===//
// AttrDef
//===----------------------------------------------------------------------===//

/// This class represents a wrapper around a tablegen AttrDef record.
class AttrDef : public AttrOrTypeDef {
public:
  using AttrOrTypeDef::AttrOrTypeDef;

  /// Returns the attributes value type builder code block, or None if it
  /// doesn't have one.
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

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_ATTRORTYPEDEF_H
