//===- AttrOrTypeDefGen.h - AttrDef/TypeDef code generator ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares classes and functions for generating C++ definitions and
// declarations for MLIR attribute and type definitions from TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_ATTRORTYPEDEFGEN_H
#define MLIR_TABLEGEN_GENERATORS_ATTRORTYPEDEFGEN_H

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Interfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <vector>

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// AttrOrTypeDefEmitter
//===----------------------------------------------------------------------===//

/// Generates C++ class declarations and definitions for a single
/// attribute or type definition derived from an AttrOrTypeDef TableGen record.
class AttrOrTypeDefEmitter {
public:
  /// Create the attribute or type class. If fatalOnError is true, assembly
  /// format parse failures are reported as fatal errors.
  AttrOrTypeDefEmitter(const AttrOrTypeDef &def, bool fatalOnError = true);

  virtual ~AttrOrTypeDefEmitter() = default;

  void emitDecl(llvm::raw_ostream &os) const;
  void emitDef(llvm::raw_ostream &os) const;

protected:
  /// Add traits from the TableGen definition to the class.
  virtual void createParentWithTraits();
  /// Emit top-level declarations: using declarations and any extra class
  /// declarations.
  virtual void emitTopLevelDeclarations();
  /// Emit the function that returns the type or attribute name.
  virtual void emitName();
  /// Emit the dialect name as a static member variable.
  virtual void emitDialectName();
  /// Emit attribute or type builders.
  virtual void emitBuilders();
  /// Emit a verifier declaration for custom verification (impl. provided by
  /// the users).
  virtual void emitVerifierDecl();
  /// Emit a verifier that checks type constraints.
  virtual void emitInvariantsVerifierImpl();
  /// Emit an entry point for verification that calls the invariants and
  /// custom verifier.
  virtual void emitInvariantsVerifier(bool hasImpl, bool hasCustomVerifier);
  /// Emit parsers and printers.
  virtual void emitParserPrinter();
  /// Emit parameter accessors, if required.
  virtual void emitAccessors();
  /// Emit interface methods.
  virtual void emitInterfaceMethods();

  //===--------------------------------------------------------------------===//
  // Builder Emission

  /// Emit the default builder `Attribute::get`.
  virtual void emitDefaultBuilder();
  /// Emit the checked builder `Attribute::getChecked`.
  virtual void emitCheckedBuilder();
  /// Emit a custom builder.
  virtual void emitCustomBuilder(const AttrOrTypeBuilder &builder);
  /// Emit a checked custom builder.
  virtual void emitCheckedCustomBuilder(const AttrOrTypeBuilder &builder);

  //===--------------------------------------------------------------------===//
  // Interface Method Emission

  /// Emit methods for a trait.
  virtual void emitTraitMethods(const InterfaceTrait &trait);
  /// Emit a trait method.
  virtual void emitTraitMethod(const InterfaceMethod &method);
  /// Generate a using declaration for a trait method.
  virtual void genTraitMethodUsingDecl(const InterfaceTrait &trait,
                                       const InterfaceMethod &method);

  //===--------------------------------------------------------------------===//
  // OpAsm{Type,Attr}Interface Default Method Emission

  /// Emit 'getAlias' method using mnemonic as alias.
  virtual void emitMnemonicAliasMethod();

  //===--------------------------------------------------------------------===//
  // Storage Class Emission
  virtual void emitStorageClass();
  /// Generate the storage class constructor.
  virtual void emitStorageConstructor();
  /// Emit the key type `KeyTy`.
  virtual void emitKeyType();
  /// Emit the equality comparison operator.
  virtual void emitEquals();
  /// Emit the key hash function.
  virtual void emitHashKey();
  /// Emit the function to construct the storage class.
  virtual void emitConstruct();

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

  /// The set of using declarations for trait methods.
  llvm::StringSet<> interfaceUsingNames;

  /// Whether assembly format parse failures are fatal errors.
  bool fatalOnError;
};

//===----------------------------------------------------------------------===//
// AttrTypeDefGenerator
//===----------------------------------------------------------------------===//

/// Base generator for processing TableGen attr/type definitions.
class AttrTypeDefGenerator {
public:
  virtual ~AttrTypeDefGenerator() = default;

  virtual bool emitDecls(llvm::StringRef selectedDialect);
  virtual bool emitDefs(llvm::StringRef selectedDialect);

protected:
  AttrTypeDefGenerator(llvm::ArrayRef<const llvm::Record *> defs,
                       llvm::raw_ostream &os, llvm::StringRef defType,
                       llvm::StringRef valueType, bool isAttrGenerator,
                       bool fatalOnError = true);

  /// Emit the list of def type names.
  virtual void emitTypeDefList(llvm::ArrayRef<AttrOrTypeDef> defs);
  /// Emit the code to dispatch between different defs during parsing/printing.
  virtual void emitParsePrintDispatch(llvm::ArrayRef<AttrOrTypeDef> defs);

  /// The set of def records to emit.
  std::vector<const llvm::Record *> defRecords;
  /// The stream to emit to.
  llvm::raw_ostream &os;
  /// The prefix of the tablegen def name, e.g. Attr or Type.
  llvm::StringRef defType;
  /// The C++ base value type of the def, e.g. Attribute or Type.
  llvm::StringRef valueType;
  /// Flag indicating if this generator is for Attributes. False if the
  /// generator is for types.
  bool isAttrGenerator;
  /// Whether assembly format parse failures are fatal errors.
  bool fatalOnError;
};

/// A specialized generator for AttrDefs.
struct AttrDefGenerator : public AttrTypeDefGenerator {
  AttrDefGenerator(const llvm::RecordKeeper &records, llvm::raw_ostream &os,
                   bool fatalOnError = true);
};

/// A specialized generator for TypeDefs.
struct TypeDefGenerator : public AttrTypeDefGenerator {
  TypeDefGenerator(const llvm::RecordKeeper &records, llvm::raw_ostream &os,
                   bool fatalOnError = true);
};

//===----------------------------------------------------------------------===//
// Constraint Functions
//===----------------------------------------------------------------------===//

/// Emit declarations for all type constraints in records that have a C++
/// function name set.
void emitTypeConstraintDecls(const llvm::RecordKeeper &records,
                             llvm::raw_ostream &os);

/// Emit declarations for all attribute constraints in records that have a
/// C++ function name set.
void emitAttrConstraintDecls(const llvm::RecordKeeper &records,
                             llvm::raw_ostream &os);

/// Emit definitions for all type constraints in records that have a C++
/// function name set.
void emitTypeConstraintDefs(const llvm::RecordKeeper &records,
                            llvm::raw_ostream &os);

/// Emit definitions for all attribute constraints in records that have a
/// C++ function name set.
void emitAttrConstraintDefs(const llvm::RecordKeeper &records,
                            llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_ATTRORTYPEDEFGEN_H
