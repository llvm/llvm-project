//===- OpFormatGen.h - MLIR operation format generator ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for generating parsers and printers from the
// declarative format.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_OPFORMATGEN_H
#define MLIR_TABLEGEN_GENERATORS_OPFORMATGEN_H

#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Generators/FormatGen.h"
#include "mlir/TableGen/Generators/OpClass.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Property.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include <optional>
#include <vector>

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// OperationFormat
//===----------------------------------------------------------------------===//

/// Holds the parsed assembly format for an operation and drives generation of
/// the corresponding parser and printer methods.
struct OperationFormat {
  using ConstArgument =
      llvm::PointerUnion<const NamedAttribute *, const NamedTypeConstraint *>;

  /// Represents a specific resolver for an operand or result type.
  class TypeResolution {
  public:
    TypeResolution() = default;

    /// Get the index into the buildable types for this type, or std::nullopt.
    std::optional<int> getBuilderIdx() const { return builderIdx; }
    void setBuilderIdx(int idx) { builderIdx = idx; }

    /// Get the variable this type is resolved to, or nullptr.
    const NamedTypeConstraint *getVariable() const {
      return llvm::dyn_cast_if_present<const NamedTypeConstraint *>(resolver);
    }
    /// Get the attribute this type is resolved to, or nullptr.
    const NamedAttribute *getAttribute() const {
      return llvm::dyn_cast_if_present<const NamedAttribute *>(resolver);
    }
    /// Get the transformer for the type of the variable, or std::nullopt.
    std::optional<llvm::StringRef> getVarTransformer() const {
      return variableTransformer;
    }
    void setResolver(ConstArgument arg,
                     std::optional<llvm::StringRef> transformer) {
      resolver = arg;
      variableTransformer = transformer;
      assert(getVariable() || getAttribute());
    }

  private:
    /// If the type is resolved with a buildable type, this is the index into
    /// 'buildableTypes' in the parent format.
    std::optional<int> builderIdx;
    /// If the type is resolved based upon another operand or result, this is
    /// the variable or the attribute that this type is resolved to.
    ConstArgument resolver;
    /// If the type is resolved based upon another operand or result, this is
    /// a transformer to apply to the variable when resolving.
    std::optional<llvm::StringRef> variableTransformer;
  };

  /// The context in which an element is generated.
  enum class GenContext {
    /// The element is generated at the top-level or with the same behaviour.
    Normal,
    /// The element is generated inside an optional group.
    Optional
  };

  OperationFormat(const Operator &op, bool hasProperties);
  virtual ~OperationFormat() = default;

  /// Generate the operation parser from this format.
  virtual void genParser(Operator &op, OpClass &opClass);
  /// Generate the parser code for a specific format element.
  void genElementParser(FormatElement *element, MethodBody &body,
                        FmtContext &attrTypeCtx,
                        GenContext genCtx = GenContext::Normal);
  /// Generate the C++ to resolve the types of operands and results during
  /// parsing.
  virtual void genParserTypeResolution(Operator &op, MethodBody &body);
  /// Generate the C++ to resolve the types of the operands during parsing.
  virtual void genParserOperandTypeResolution(
      Operator &op, MethodBody &body,
      llvm::function_ref<void(TypeResolution &, llvm::StringRef)>
          emitTypeResolver);
  /// Generate the C++ to resolve regions during parsing.
  virtual void genParserRegionResolution(Operator &op, MethodBody &body);
  /// Generate the C++ to resolve successors during parsing.
  virtual void genParserSuccessorResolution(Operator &op, MethodBody &body);
  /// Generate the C++ to handle variadic segment size traits.
  virtual void genParserVariadicSegmentResolution(Operator &op,
                                                  MethodBody &body);

  /// Generate the operation printer from this format.
  virtual void genPrinter(Operator &op, OpClass &opClass);
  /// Generate the printer code for a specific format element.
  virtual void genElementPrinter(FormatElement *element, MethodBody &body,
                                 Operator &op, bool &shouldEmitSpace,
                                 bool &lastWasPunctuation);

  /// The various elements in this format.
  std::vector<FormatElement *> elements;

  /// A flag indicating if all operand/result types were seen. If the format
  /// contains these, it cannot contain individual type resolvers.
  bool allOperands = false, allOperandTypes = false, allResultTypes = false;

  /// A flag indicating if this operation infers its result types.
  bool infersResultTypes = false;

  /// A flag indicating if this operation has the SingleBlockImplicitTerminator
  /// trait.
  bool hasImplicitTermTrait;

  /// A flag indicating if this operation has the SingleBlock trait.
  bool hasSingleBlockTrait;

  /// Indicate whether we need to use properties for the current operator.
  bool useProperties;

  /// Indicate whether prop-dict is used in the format.
  bool hasPropDict;

  /// The Operation class name.
  llvm::StringRef opCppClassName;

  /// A map of buildable types to indices.
  llvm::MapVector<llvm::StringRef, int, llvm::StringMap<int>> buildableTypes;

  /// The index of the buildable type, if valid, for every operand and result.
  std::vector<TypeResolution> operandTypes, resultTypes;

  /// The set of attributes explicitly used within the format.
  llvm::SmallSetVector<const NamedAttribute *, 8> usedAttributes;
  llvm::StringSet<> inferredAttributes;

  /// The set of properties explicitly used within the format.
  llvm::SmallSetVector<const NamedProperty *, 8> usedProperties;
};

//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

/// Generate the assembly format for the given operator. If fatalOnError is
/// true, format parse errors cause the process to exit; otherwise they are
/// silently ignored.
void generateOpFormat(const Operator &constOp, OpClass &opClass,
                      bool hasProperties, bool fatalOnError = true);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_OPFORMATGEN_H
