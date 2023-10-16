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

  let hasOperationAttrVerify = 1;
  let extraClassDeclaration = [{
    /// Symbol name for the default entry point "named sequence".
    constexpr const static ::llvm::StringLiteral
        kTransformEntryPointSymbolName = "__transform_main";
    
    /// Name of the attribute attachable to the symbol table operation
    /// containing named sequences. This is used to trigger verification.
    constexpr const static ::llvm::StringLiteral
        kWithNamedSequenceAttrName = "transform.with_named_sequence";

    /// Name of the attribute attachable to an operation so it can be
    /// identified as root by the default interpreter pass.
    constexpr const static ::llvm::StringLiteral kTargetTagAttrName =
        "transform.target_tag";

    /// Name of the attribute attachable to an operation, indicating that
    /// TrackingListener failures should be silenced.
    constexpr const static ::llvm::StringLiteral
        kSilenceTrackingFailuresAttrName =
            "transform.silence_tracking_failures";

    /// Names of the attributes indicating whether an argument of an external
    /// transform dialect symbol is consumed or only read.
    constexpr const static ::llvm::StringLiteral kArgConsumedAttrName =
        "transform.consumed";
    constexpr const static ::llvm::StringLiteral kArgReadOnlyAttrName =
        "transform.readonly";

    /// Names of the attributes indicating whether an argument of an external
    /// transform dialect symbol is consumed or only read.
    StringAttr getConsumedAttrName() const {
      return StringAttr::get(getContext(), kArgConsumedAttrName);
    }
    StringAttr getReadOnlyAttrName() const {
      return StringAttr::get(getContext(), kArgReadOnlyAttrName);
    }

    template <typename DataTy>
    const DataTy &getExtraData() const {
      return *static_cast<const DataTy *>(
          extraData.at(::mlir::TypeID::get<DataTy>()).get());
    }

    /// Parses a type registered by this dialect or one of its extensions.
    ::mlir::Type parseType(::mlir::DialectAsmParser & parser) const override;

    /// Prints a type registered by this dialect or one of its extensions.
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter & printer)
        const override;

    /// Parser callback for an individual type registered by this dialect or
    /// its extensions.
    using ExtensionTypeParsingHook = ::mlir::Type (*)(::mlir::AsmParser &);

    /// Printer callback for an individual type registered by this dialect or
    /// its extensions.
    using ExtensionTypePrintingHook =
        std::function<void(::mlir::Type, ::mlir::AsmPrinter &)>;

    /// Loads the given module into the transform symbol library module.
    void initializeLibraryModule();

    /// Loads the given module into the transform symbol library module.
    LogicalResult loadIntoLibraryModule(::mlir::OwningOpRef<::mlir::ModuleOp> &&
                                        library);

    /// Returns the transform symbol library module available to all dialect
    /// users.
    ModuleOp getLibraryModule() const {
      if (libraryModule)
        return libraryModule.get();
      return ModuleOp();
    }

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

    /// Registers types specified as template parameters with the Transform
    /// dialect. Checks that they meet the requirements for Transform IR types.
    template <typename... TypeTys>
    void addTypesChecked() {
      (addTypeIfNotRegistered<TypeTys>(), ...);
    }
    template <typename Type>
    void addTypeIfNotRegistered();

    /// Reports a repeated registration error of a type with the given
    /// mnemonic.
    [[noreturn]] void reportDuplicateTypeRegistration(StringRef mnemonic);

    /// Registers dialect types with the context.
    void initializeTypes();

    // Give extensions access to injection functions.
    template <typename, typename...>
    friend class TransformDialectExtension;

    /// Gets a mutable reference to extra data of the kind specified as
    /// template argument. Allocates the data on the first call.
    template <typename DataTy>
    DataTy &getOrCreateExtraData();

    //===----------------------------------------------------------------===//
    // Data fields
    //===----------------------------------------------------------------===//

    /// Additional data associated with and owned by the dialect. Accessible
    /// to extensions.
    ::llvm::DenseMap<
        ::mlir::TypeID,
        std::unique_ptr<::mlir::transform::detail::TransformDialectDataBase>>
        extraData;

    /// A map from type mnemonic to its parsing function for the remainder of
    /// the syntax. The parser has access to the mnemonic, so it is used for
    /// further dispatch.
    ::llvm::StringMap<ExtensionTypeParsingHook> typeParsingHooks;

    /// A map from type TypeID to its printing function. No need to do string
    /// lookups when the type is fully constructed.
    ::llvm::DenseMap<::mlir::TypeID, ExtensionTypePrintingHook>
        typePrintingHooks;

    /// Module containing symbols, e.g. named sequences, that will be resolved
    /// by the interpreter when used.
    ::mlir::OwningOpRef<::mlir::ModuleOp> libraryModule;
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
