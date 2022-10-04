//===- TransformDialect.h - Transform Dialect Definition --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace transform {
#ifndef NDEBUG
namespace detail {
/// Asserts that the operations provided as template arguments implement the
/// TransformOpInterface and MemoryEffectsOpInterface. This must be a dynamic
/// assertion since interface implementations may be registered at runtime.
template <typename OpTy>
static inline void checkImplementsTransformInterface(MLIRContext *context) {
  // Since the operation is being inserted into the Transform dialect and the
  // dialect does not implement the interface fallback, only check for the op
  // itself having the interface implementation.
  RegisteredOperationName opName =
      *RegisteredOperationName::lookup(OpTy::getOperationName(), context);
  assert((opName.hasInterface<TransformOpInterface>() ||
          opName.hasTrait<OpTrait::IsTerminator>()) &&
         "non-terminator ops injected into the transform dialect must "
         "implement TransformOpInterface");
  assert(opName.hasInterface<MemoryEffectOpInterface>() &&
         "ops injected into the transform dialect must implement "
         "MemoryEffectsOpInterface");
}

/// Asserts that the type provided as template argument implements the
/// TransformTypeInterface. This must be a dynamic assertion since interface
/// implementations may be registered at runtime.
void checkImplementsTransformTypeInterface(TypeID typeID, MLIRContext *context);
} // namespace detail
#endif // NDEBUG
} // namespace transform
} // namespace mlir

#include "mlir/Dialect/Transform/IR/TransformDialect.h.inc"

namespace mlir {
namespace transform {

/// Base class for extensions of the Transform dialect that supports injecting
/// operations into the Transform dialect at load time. Concrete extensions are
/// expected to derive this class and register operations in the constructor.
/// They can be registered with the DialectRegistry and automatically applied
/// to the Transform dialect when it is loaded.
///
/// Derived classes are expected to define a `void init()` function in which
/// they can call various protected methods of the base class to register
/// extension operations and declare their dependencies.
///
/// By default, the extension is configured both for construction of the
/// Transform IR and for its application to some payload. If only the
/// construction is desired, the extension can be switched to "build-only" mode
/// that avoids loading the dialects that are only necessary for transforming
/// the payload. To perform the switch, the extension must be wrapped into the
/// `BuildOnly` class template (see below) when it is registered, as in:
///
///    dialectRegistry.addExtension<BuildOnly<MyTransformDialectExt>>();
///
/// instead of:
///
///    dialectRegistry.addExtension<MyTransformDialectExt>();
///
/// Derived classes must reexport the constructor of this class or otherwise
/// forward its boolean argument to support this behavior.
template <typename DerivedTy, typename... ExtraDialects>
class TransformDialectExtension
    : public DialectExtension<DerivedTy, TransformDialect, ExtraDialects...> {
  using Initializer = std::function<void(TransformDialect *)>;
  using DialectLoader = std::function<void(MLIRContext *)>;

public:
  /// Extension application hook. Actually loads the dependent dialects and
  /// registers the additional operations. Not expected to be called directly.
  void apply(MLIRContext *context, TransformDialect *transformDialect,
             ExtraDialects *...) const final {
    for (const DialectLoader &loader : dialectLoaders)
      loader(context);

    // Only load generated dialects if the user intends to apply
    // transformations specified by the extension.
    if (!buildOnly)
      for (const DialectLoader &loader : generatedDialectLoaders)
        loader(context);

    for (const Initializer &init : opInitializers)
      init(transformDialect);
    transformDialect->mergeInPDLMatchHooks(std::move(pdlMatchConstraintFns));
  }

protected:
  using Base = TransformDialectExtension<DerivedTy, ExtraDialects...>;

  /// Extension constructor. The argument indicates whether to skip generated
  /// dialects when applying the extension.
  explicit TransformDialectExtension(bool buildOnly = false)
      : buildOnly(buildOnly) {
    static_cast<DerivedTy *>(this)->init();
  }

  /// Hook for derived classes to inject constructor behavior.
  void init() {}

  /// Injects the operations into the Transform dialect. The operations must
  /// implement the TransformOpInterface and MemoryEffectsOpInterface, and the
  /// implementations must be already available when the operation is injected.
  template <typename... OpTys>
  void registerTransformOps() {
    opInitializers.push_back([](TransformDialect *transformDialect) {
      transformDialect->addOperationsChecked<OpTys...>();
    });
  }

  /// Injects the types into the Transform dialect. The types must implement
  /// the TransformTypeInterface and the implementation must be already
  /// available when the type is injected. Furthermore, the types must provide
  /// a `getMnemonic` static method returning an object convertible to
  /// `StringRef` that is unique across all injected types.
  template <typename... TypeTys>
  void registerTypes() {
    opInitializers.push_back([](TransformDialect *transformDialect) {
      transformDialect->addTypesChecked<TypeTys...>();
    });
  }

  /// Declares that this Transform dialect extension depends on the dialect
  /// provided as template parameter. When the Transform dialect is loaded,
  /// dependent dialects will be loaded as well. This is intended for dialects
  /// that contain attributes and types used in creation and canonicalization of
  /// the injected operations, similarly to how the dialect definition may list
  /// dependent dialects. This is *not* intended for dialects entities from
  /// which may be produced when applying the transformations specified by ops
  /// registered by this extension.
  template <typename DialectTy>
  void declareDependentDialect() {
    dialectLoaders.push_back(
        [](MLIRContext *context) { context->loadDialect<DialectTy>(); });
  }

  /// Declares that the transformations associated with the operations
  /// registered by this dialect extension may produce operations from the
  /// dialect provided as template parameter while processing payload IR that
  /// does not contain the operations from said dialect. This is similar to
  /// dependent dialects of a pass. These dialects will be loaded along with the
  /// transform dialect unless the extension is in the build-only mode.
  template <typename DialectTy>
  void declareGeneratedDialect() {
    generatedDialectLoaders.push_back(
        [](MLIRContext *context) { context->loadDialect<DialectTy>(); });
  }

  /// Injects the named constraint to make it available for use with the
  /// PDLMatchOp in the transform dialect.
  void registerPDLMatchConstraintFn(StringRef name,
                                    PDLConstraintFunction &&fn) {
    pdlMatchConstraintFns.try_emplace(name,
                                      std::forward<PDLConstraintFunction>(fn));
  }
  template <typename ConstraintFnTy>
  void registerPDLMatchConstraintFn(StringRef name, ConstraintFnTy &&fn) {
    pdlMatchConstraintFns.try_emplace(
        name, ::mlir::detail::pdl_function_builder::buildConstraintFn(
                  std::forward<ConstraintFnTy>(fn)));
  }

private:
  SmallVector<Initializer> opInitializers;

  /// Callbacks loading the dependent dialects, i.e. the dialect needed for the
  /// extension ops.
  SmallVector<DialectLoader> dialectLoaders;

  /// Callbacks loading the generated dialects, i.e. the dialects produced when
  /// applying the transformations.
  SmallVector<DialectLoader> generatedDialectLoaders;

  /// A list of constraints that should be made available to PDL patterns
  /// processed by PDLMatchOp in the Transform dialect.
  ///
  /// Declared as mutable so its contents can be moved in the `apply` const
  /// method, which is only called once.
  mutable llvm::StringMap<PDLConstraintFunction> pdlMatchConstraintFns;

  /// Indicates that the extension is in build-only mode.
  bool buildOnly;
};

template <typename Type>
void TransformDialect::addTypeIfNotRegistered() {
  // Use the address of the parse method as a proxy for identifying whether we
  // are registering the same type class for the same mnemonic.
  StringRef mnemonic = Type::getMnemonic();
  auto [it, inserted] = typeParsingHooks.try_emplace(mnemonic, Type::parse);
  if (!inserted) {
    const ExtensionTypeParsingHook &parsingHook = it->getValue();
    if (*parsingHook.target<mlir::Type (*)(AsmParser &)>() != &Type::parse)
      reportDuplicateTypeRegistration(mnemonic);
  }
  typePrintingHooks.try_emplace(
      TypeID::get<Type>(), +[](mlir::Type type, AsmPrinter &printer) {
        printer << Type::getMnemonic();
        cast<Type>(type).print(printer);
      });
  addTypes<Type>();
}

/// A wrapper for transform dialect extensions that forces them to be
/// constructed in the build-only mode.
template <typename DerivedTy>
class BuildOnly : public DerivedTy {
public:
  BuildOnly() : DerivedTy(/*buildOnly=*/true) {}
};

} // namespace transform
} // namespace mlir

#include "mlir/Dialect/Transform/IR/TransformDialectEnums.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT_H
