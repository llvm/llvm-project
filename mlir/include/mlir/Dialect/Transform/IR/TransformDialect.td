//===- TransformDialect.td - Transform dialect definition --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT

include "mlir/IR/OpBase.td"

def Transform_Dialect : Dialect {
  let summary = "Fine-grain transformation control dialect";
  // For description, see docs/Dialects/Transform.md.

  let name = "transform";
  let cppNamespace = "::mlir::transform";

  let dependentDialects = [
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect",
  ];

  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
      /// Name of the attribute attachable to the symbol table operation
      /// containing named sequences. This is used to trigger verification.
      constexpr const static llvm::StringLiteral
          kWithNamedSequenceAttrName = "transform.with_named_sequence";

      /// Names of the attribute attachable to an operation so it can be
      /// identified as root by the default interpreter pass.
      constexpr const static llvm::StringLiteral
          kTargetTagAttrName = "transform.target_tag";

      /// Names of the attributes indicating whether an argument of an external
      /// transform dialect symbol is consumed or only read.
      constexpr const static llvm::StringLiteral
          kArgConsumedAttrName = "transform.consumed";
      constexpr const static llvm::StringLiteral
          kArgReadOnlyAttrName = "transform.readonly";

      /// Returns the named PDL constraint functions available in the dialect
      /// as a map from their name to the function.
      const ::llvm::StringMap<::mlir::PDLConstraintFunction> &
      getPDLConstraintHooks() const;

      /// Parses a type registered by this dialect or one of its extensions.
      ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

      /// Prints a type registered by this dialect or one of its extensions.
      void printType(::mlir::Type type,
                     ::mlir::DialectAsmPrinter &printer) const override;

      /// Parser callback for an individual type registered by this dialect or
      /// its extensions.
      using ExtensionTypeParsingHook = ::mlir::Type (*)(::mlir::AsmParser &);

      /// Printer callback for an individual type registered by this dialect or
      /// its extensions.
      using ExtensionTypePrintingHook =
          std::function<void (::mlir::Type, ::mlir::AsmPrinter &)>;

    private:
      /// Registers operations specified as template parameters with this
      /// dialect. Checks that they implement the required interfaces.
      template <typename... OpTys>
      void addOperationsChecked() {
        (addOperationIfNotRegistered<OpTys>(), ...);
      }
      template <typename OpTy>
      void addOperationIfNotRegistered();

      /// Reports a repeated registration error of an op with the given name.
      [[noreturn]] void reportDuplicateOpRegistration(StringRef opName);

      /// Registers the types specified as template parameters with the
      /// Transform dialect. Checks that they meet the requirements for
      /// Transform IR types.
      template <typename... TypeTys>
      void addTypesChecked() {
        (addTypeIfNotRegistered<TypeTys>(), ...);
      }
      template <typename Type>
      void addTypeIfNotRegistered();

      /// Reports a repeated registration error of a type with the given
      /// mnemonic.
      [[noreturn]] void reportDuplicateTypeRegistration(StringRef mnemonic);

      void initializeTypes();

      template <typename, typename...>
      friend class TransformDialectExtension;

      /// Takes ownership of the named PDL constraint function from the given
      /// map and makes them available for use by the operations in the dialect.
      void mergeInPDLMatchHooks(
          ::llvm::StringMap<::mlir::PDLConstraintFunction> &&constraintFns);

      //===----------------------------------------------------------------===//
      // Data fields
      //===----------------------------------------------------------------===//

      /// A container for PDL constraint function that can be used by
      /// operations in this dialect.
      ::mlir::PDLPatternModule pdlMatchHooks;

      /// A map from type mnemonic to its parsing function for the remainder of
      /// the syntax. The parser has access to the mnemonic, so it is used for
      /// further dispatch.
      ::llvm::StringMap<ExtensionTypeParsingHook> typeParsingHooks;

      /// A map from type TypeID to its printing function. No need to do string
      /// lookups when the type is fully constructed.
      ::llvm::DenseMap<::mlir::TypeID, ExtensionTypePrintingHook>
      typePrintingHooks;
  }];
}

// Base class for ops that belong to the transform dialect. Ops defined in
// extensions of this dialect may also use this.
class TransformDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<Transform_Dialect, mnemonic, traits>;

// Trait for operations that may be top-level operations in Transform IR.
// Operations must have one single-block region and must be usable without
// operands. See the C++ definition of the trait for more information.
def PossibleTopLevelTransformOpTrait
    : NativeOpTrait<"PossibleTopLevelTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT
