//===- OpDefinitionsGen.h - Op definitions generator -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_OPDEFINITIONSGEN_H
#define MLIR_TABLEGEN_GENERATORS_OPDEFINITIONSGEN_H

#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Generators/OpAdaptorHelper.h"
#include "mlir/TableGen/Generators/OpClass.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Property.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// OpEmitter
//===----------------------------------------------------------------------===//

/// Generates C++ declarations and definitions for a single operation record.
class OpEmitter {
public:
  using ConstArgument =
      llvm::PointerUnion<const AttributeMetadata *, const NamedProperty *>;

  /// Emit C++ declarations for op.
  static void
  emitDecl(const Operator &op, llvm::raw_ostream &os,
           const StaticVerifierFunctionEmitter &staticVerifierEmitter,
           bool fatalOnError = true);
  /// Emit C++ definitions for op.
  static void
  emitDef(const Operator &op, llvm::raw_ostream &os,
          const StaticVerifierFunctionEmitter &staticVerifierEmitter,
          bool fatalOnError = true);

  virtual ~OpEmitter() = default;

protected:
  OpEmitter(const Operator &op,
            const StaticVerifierFunctionEmitter &staticVerifierEmitter,
            bool fatalOnError = true);

  void emitDecl(llvm::raw_ostream &os);
  void emitDef(llvm::raw_ostream &os);

  /// Generate methods for accessing the attribute names of this operation.
  virtual void genAttrNameGetters();

  /// Generate the OpAsmOpInterface for this operation if possible.
  virtual void genOpAsmInterface();

  /// Generate the getOperationName method for this op.
  virtual void genOpNameGetter();

  /// Generate code to manage the properties, if any.
  virtual void genPropertiesSupport();

  /// Generate code to manage the encoding of properties to bytecode.
  virtual void genPropertiesSupportForBytecode(
      llvm::ArrayRef<ConstArgument> attrOrProperties);

  /// Generate getters for the properties.
  virtual void genPropGetters();

  /// Generate setters for the properties.
  virtual void genPropSetters();

  /// Generate getters for the attributes.
  virtual void genAttrGetters();

  /// Generate setters for the attributes.
  virtual void genAttrSetters();

  /// Generate removers for optional attributes.
  virtual void genOptionalAttrRemovers();

  /// Generate getters for named operands.
  virtual void genNamedOperandGetters();

  /// Generate setters for named operands.
  virtual void genNamedOperandSetters();

  /// Generate getters for named results.
  virtual void genNamedResultGetters();

  /// Generate getters for named regions.
  virtual void genNamedRegionGetters();

  /// Generate getters for named successors.
  virtual void genNamedSuccessorGetters();

  /// Generate the method to populate default attributes.
  virtual void genPopulateDefaultAttributes();

  /// Generate builder methods for the operation.
  virtual void genBuilder();

  /// Generate the build() method that takes each operand/attribute as a
  /// stand-alone parameter.
  virtual void genSeparateArgParamBuilder();
  virtual void
  genInlineCreateBody(const SmallVector<MethodParameter> &paramList);

  /// Generate the build() method that uses the first operand's type as all
  /// results' types, with stand-alone parameters.
  virtual void genUseOperandAsResultTypeSeparateParamBuilder();

  /// The kind of collective builder to generate.
  enum class CollectiveBuilderKind {
    /// Inherent attributes/properties are passed by const Properties&.
    PropStruct,
    /// Inherent attributes/properties are passed by attribute dictionary.
    AttrDict,
  };

  /// Generate the build() method that uses the first operand's type as all
  /// results' types, with collective parameters.
  virtual void
  genUseOperandAsResultTypeCollectiveParamBuilder(CollectiveBuilderKind kind);

  /// Generate the build() method that uses inferred types as result types.
  /// Requires InferTypeOpInterface.
  virtual void
  genInferredTypeCollectiveParamBuilder(CollectiveBuilderKind kind);

  /// Generate the build() method that uses the first attribute's type as all
  /// result types, with collective parameters.
  virtual void
  genUseAttrAsResultTypeCollectiveParamBuilder(CollectiveBuilderKind kind);

  /// Generate the build() method with collective result-type and
  /// operand/attribute parameters.
  virtual void genCollectiveParamBuilder(CollectiveBuilderKind kind);

  /// The kind of parameter to generate for result types in builders.
  enum class TypeParamKind {
    /// No result type in the parameter list.
    None,
    /// A separate parameter for each result type.
    Separate,
    /// An ArrayRef<Type> for all result types.
    Collective,
  };

  /// The kind of parameter to generate for attributes in builders.
  enum class AttrParamKind {
    /// A wrapped MLIR Attribute instance.
    WrappedAttr,
    /// A raw value without MLIR Attribute wrapper.
    UnwrappedValue,
  };

  /// Build the parameter list for a build() method. Writes to paramList and
  /// updates resultTypeNames. inferredAttributes is populated with attributes
  /// elided from the build list. typeParamKind and attrParamKind control how
  /// result types and attributes are placed in the parameter list.
  virtual void
  buildParamList(SmallVectorImpl<MethodParameter> &paramList,
                 llvm::StringSet<> &inferredAttributes,
                 SmallVectorImpl<std::string> &resultTypeNames,
                 TypeParamKind typeParamKind,
                 AttrParamKind attrParamKind = AttrParamKind::WrappedAttr);

  /// Add op arguments and regions into the operation state for build() methods.
  virtual void
  genCodeForAddingArgAndRegionForBuilder(MethodBody &body,
                                         llvm::StringSet<> &inferredAttributes,
                                         bool isRawValueAttr = false);

  /// Generate canonicalizer declarations for the operation.
  virtual void genCanonicalizerDecls();

  /// Generate the folder declaration for the operation.
  virtual void genFolderDecls();

  /// Generate the parser for the operation.
  virtual void genParser();

  /// Generate the printer for the operation.
  virtual void genPrinter();

  /// Generate the verify method for the operation.
  virtual void genVerifier();

  /// Generate custom verify methods for the operation.
  virtual void genCustomVerifier();

  /// Generate verify statements for operands and results. The generated code
  /// is attached to body.
  virtual void genOperandResultVerifier(MethodBody &body,
                                        Operator::const_value_range values,
                                        llvm::StringRef valueKind);

  /// Generate verify statements for regions. The generated code is attached to
  /// body.
  virtual void genRegionVerifier(MethodBody &body);

  /// Generate verify statements for successors. The generated code is attached
  /// to body.
  virtual void genSuccessorVerifier(MethodBody &body);

  /// Generate the traits used by the object.
  virtual void genTraits();

  /// Generate OpInterface methods for all interfaces.
  virtual void genOpInterfaceMethods();

  /// Generate OpInterface methods for the given interface.
  virtual void genOpInterfaceMethods(const tblgen::InterfaceTrait *trait);

  /// Generate an op interface method for the given interface method. If
  /// declaration is true, generates a declaration, else a definition.
  virtual Method *genOpInterfaceMethod(const tblgen::InterfaceMethod &method,
                                       bool declaration = true);

  /// Generate a using declaration for an op interface method to include the
  /// default implementation from the interface trait. This is needed when the
  /// interface defines multiple methods with the same name but some have a
  /// default implementation and some don't.
  virtual UsingDeclaration *
  genOpInterfaceMethodUsingDecl(const tblgen::InterfaceTrait *opTrait,
                                const tblgen::InterfaceMethod &method);

  /// Generate the side-effect interface methods.
  virtual void genSideEffectInterfaceMethods();

  /// Generate the type inference interface methods.
  virtual void genTypeInterfaceMethods();

  // The TableGen record for this op.
  // TODO: OpEmitter should not have a Record directly,
  // it should rather go through the Operator for better abstraction.
  const llvm::Record &def;

  // The wrapper operator class for querying information from this op.
  const Operator &op;

  // The C++ code builder for this op.
  OpClass opClass;

  // The format context for verification code generation.
  FmtContext verifyCtx;

  // The emitter containing all of the locally emitted verification functions.
  const StaticVerifierFunctionEmitter &staticVerifierEmitter;

  // Helper for emitting op code.
  OpOrAdaptorHelper emitHelper;

  // Keep track of interface using declarations generated to avoid duplicates.
  llvm::StringSet<> interfaceUsingNames;

  // Whether to emit fatal errors or not.
  bool fatalOnError;
};

//===----------------------------------------------------------------------===//
// Top-level entry points
//===----------------------------------------------------------------------===//

/// Emit op declarations for all op records in defs. If fatalOnError is
/// true, assembly format parse errors are fatal; otherwise they are ignored.
bool emitOpDecls(const llvm::RecordKeeper &records,
                 llvm::ArrayRef<const llvm::Record *> defs, unsigned shardCount,
                 llvm::raw_ostream &os, bool fatalOnError = true);

/// Generate the dialect op registration hook and op class definitions for a
/// shard of ops.
void emitOpDefShard(const llvm::RecordKeeper &records,
                    llvm::ArrayRef<const llvm::Record *> shardDefs,
                    const Dialect &dialect, unsigned shardIndex,
                    unsigned shardCount, llvm::raw_ostream &os,
                    bool fatalOnError = true);

/// Emit op definitions for all op records in defs. If fatalOnError is
/// true, assembly format parse errors are fatal; otherwise they are ignored.
bool emitOpDefs(const llvm::RecordKeeper &records,
                llvm::ArrayRef<const llvm::Record *> defs, unsigned shardCount,
                llvm::raw_ostream &os, bool fatalOnError = true);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_OPDEFINITIONSGEN_H
