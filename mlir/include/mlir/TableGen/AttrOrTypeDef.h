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
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/Trait.h"

namespace llvm {
class DagInit;
class Record;
class SMLoc;
} // namespace llvm

namespace mlir {
namespace tblgen {
class MethodParameter;
class MethodParameter;
class InterfaceMethod;
class Dialect;

//===----------------------------------------------------------------------===//
// AttrOrTypeBuilder
//===----------------------------------------------------------------------===//

/// Wrapper class that represents a Tablegen AttrOrTypeBuilder.
class AttrOrTypeBuilder : public Builder {
public:
  using Builder::Builder;

  /// Returns an optional builder return type.
  std::optional<StringRef> getReturnType() const;

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
  std::optional<StringRef> getAllocator() const;

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
  std::optional<StringRef> getParser() const;

  /// If this is a type constraint, return it.
  std::optional<Constraint> getConstraint() const;

  /// Get an optional C++ parameter printer.
  std::optional<StringRef> getPrinter() const;

  /// Get a description of this parameter for documentation purposes.
  std::optional<StringRef> getSummary() const;

  /// Get the assembly syntax documentation.
  StringRef getSyntax() const;

  /// Returns true if the parameter is optional.
  bool isOptional() const;

  /// Get the default value of the parameter if it has one.
  std::optional<StringRef> getDefaultValue() const;

  /// Return the underlying def of this parameter.
  const llvm::Init *getDef() const;

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
  std::optional<StringRef> getMnemonic() const;

  /// Returns if the attribute or type has a custom assembly format implemented
  /// in C++. Corresponds to the `hasCustomAssemblyFormat` field.
  bool hasCustomAssemblyFormat() const;

  /// Returns the custom assembly format, if one was specified.
  std::optional<StringRef> getAssemblyFormat() const;

  /// Returns true if the accessors based on the parameters should be generated.
  bool genAccessors() const;

  /// Return true if we need to generate the verify declaration and getChecked
  /// method.
  bool genVerifyDecl() const;

  /// Return true if we need to generate any type constraint verification and
  /// the getChecked method.
  bool genVerifyInvariantsImpl() const;

  /// Returns the def's extra class declaration code.
  std::optional<StringRef> getExtraDecls() const;

  /// Returns the def's extra class definition code.
  std::optional<StringRef> getExtraDefs() const;

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

  /// Returns the attributes value type builder code block, or std::nullopt if
  /// it doesn't have one.
  std::optional<StringRef> getTypeBuilder() const;

  static bool classof(const AttrOrTypeDef *def);

  /// Get the unique attribute name "dialect.attrname".
  StringRef getAttrName() const;
};

//===----------------------------------------------------------------------===//
// TypeDef
//===----------------------------------------------------------------------===//

/// This class represents a wrapper around a tablegen TypeDef record.
class TypeDef : public AttrOrTypeDef {
public:
  using AttrOrTypeDef::AttrOrTypeDef;

  static bool classof(const AttrOrTypeDef *def);

  /// Get the unique type name "dialect.typename".
  StringRef getTypeName() const;
};

class DefGen {
public:
  /// Create the attribute or type class.
  DefGen(const AttrOrTypeDef &def);

  void emitDecl(raw_ostream &os) const;
  void emitDef(raw_ostream &os) const;

private:
  /// Add traits from the TableGen definition to the class.
  void createParentWithTraits();
  /// Emit top-level declarations: using declarations and any extra class
  /// declarations.
  void emitTopLevelDeclarations();
  /// Emit the function that returns the type or attribute name.
  void emitName();
  /// Emit the dialect name as a static member variable.
  void emitDialectName();
  /// Emit attribute or type builders.
  void emitBuilders();
  /// Emit a verifier declaration for custom verification (impl. provided by
  /// the users).
  void emitVerifierDecl();
  /// Emit a verifier that checks type constraints.
  void emitInvariantsVerifierImpl();
  /// Emit an entry poiunt for verification that calls the invariants and
  /// custom verifier.
  void emitInvariantsVerifier(bool hasImpl, bool hasCustomVerifier);
  /// Emit parsers and printers.
  void emitParserPrinter();
  /// Emit parameter accessors, if required.
  void emitAccessors();
  /// Emit interface methods.
  void emitInterfaceMethods();

  //===--------------------------------------------------------------------===//
  // Builder Emission

  /// Emit the default builder `Attribute::get`
  void emitDefaultBuilder();
  /// Emit the checked builder `Attribute::getChecked`
  void emitCheckedBuilder();
  /// Emit a custom builder.
  void emitCustomBuilder(const AttrOrTypeBuilder &builder);
  /// Emit a checked custom builder.
  void emitCheckedCustomBuilder(const AttrOrTypeBuilder &builder);

  //===--------------------------------------------------------------------===//
  // Interface Method Emission

  /// Emit methods for a trait.
  void emitTraitMethods(const InterfaceTrait &trait);
  /// Emit a trait method.
  void emitTraitMethod(const InterfaceMethod &method);

  //===--------------------------------------------------------------------===//
  // Storage Class Emission
  void emitStorageClass();
  /// Generate the storage class constructor.
  void emitStorageConstructor();
  /// Emit the key type `KeyTy`.
  void emitKeyType();
  /// Emit the equality comparison operator.
  void emitEquals();
  /// Emit the key hash function.
  void emitHashKey();
  /// Emit the function to construct the storage class.
  void emitConstruct();

  //===--------------------------------------------------------------------===//
  // Utility Function Declarations

  /// Get the method parameters for a def builder, where the first several
  /// parameters may be different.
  SmallVector<MethodParameter>
  getBuilderParams(std::initializer_list<MethodParameter> prefix) const;

  //===--------------------------------------------------------------------===//
  // Class fields

  /// The attribute or type definition.
  const AttrOrTypeDef &def;
  /// The list of attribute or type parameters.
  ArrayRef<AttrOrTypeParameter> params;
  /// The attribute or type class.
  Class defCls;
  /// An optional attribute or type storage class. The storage class will
  /// exist if and only if the def has more than zero parameters.
  std::optional<Class> storageCls;

  /// The C++ base value of the def, either "Attribute" or "Type".
  StringRef valueType;
  /// The prefix/suffix of the TableGen def name, either "Attr" or "Type".
  StringRef defType;
};

class DefGenerator {
public:
  bool emitDecls(StringRef selectedDialect);
  bool emitDefs(StringRef selectedDialect);

protected:
  DefGenerator(ArrayRef<const llvm::Record *> defs, raw_ostream &os,
               StringRef defType, StringRef valueType, bool isAttrGenerator);

  /// Emit the list of def type names.
  void emitTypeDefList(ArrayRef<AttrOrTypeDef> defs);
  /// Emit the code to dispatch between different defs during parsing/printing.
  void emitParsePrintDispatch(ArrayRef<AttrOrTypeDef> defs);

  /// The set of def records to emit.
  std::vector<const llvm::Record *> defRecords;
  /// The attribute or type class to emit.
  /// The stream to emit to.
  raw_ostream &os;
  /// The prefix of the tablegen def name, e.g. Attr or Type.
  StringRef defType;
  /// The C++ base value type of the def, e.g. Attribute or Type.
  StringRef valueType;
  /// Flag indicating if this generator is for Attributes. False if the
  /// generator is for types.
  bool isAttrGenerator;
};

/// A specialized generator for AttrDefs.
struct AttrDefGenerator : public DefGenerator {
  AttrDefGenerator(const llvm::RecordKeeper &records, raw_ostream &os)
      : DefGenerator(records.getAllDerivedDefinitionsIfDefined("AttrDef"), os,
                     "Attr", "Attribute", /*isAttrGenerator=*/true) {}
};
/// A specialized generator for TypeDefs.
struct TypeDefGenerator : public DefGenerator {
  TypeDefGenerator(const llvm::RecordKeeper &records, raw_ostream &os)
      : DefGenerator(records.getAllDerivedDefinitionsIfDefined("TypeDef"), os,
                     "Type", "Type", /*isAttrGenerator=*/false) {}
};

void emitTypeConstraintDecls(const llvm::RecordKeeper &records,
                             raw_ostream &os);

void emitTypeConstraintDefs(const llvm::RecordKeeper &records, raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_ATTRORTYPEDEF_H
