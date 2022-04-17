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
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h.inc"

namespace mlir {
namespace transform {

#ifndef NDEBUG
namespace detail {
/// Asserts that the operations provided as template arguments implement the
/// TransformOpInterface. This must be a dynamic assertion since interface
/// implementations may be registered at runtime.
template <typename OpTy>
static inline void checkImplementsTransformInterface(MLIRContext *context) {
  // Since the operation is being inserted into the Transform dialect and the
  // dialect does not implement the interface fallback, only check for the op
  // itself having the interface implementation.
  RegisteredOperationName opName =
      *RegisteredOperationName::lookup(OpTy::getOperationName(), context);
  assert(opName.hasInterface<TransformOpInterface>() &&
         "ops injected into the transform dialect must implement "
         "TransformOpInterface");
}
} // namespace detail
#endif // NDEBUG

/// Base class for extensions of the Transform dialect that supports injecting
/// operations into the Transform dialect at load time. Concrete extensions are
/// expected to derive this class and register operations in the constructor.
/// They can be registered with the DialectRegistry and automatically applied
/// to the Transform dialect when it is loaded.
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
    for (const Initializer &init : opInitializers)
      init(transformDialect);
  }

protected:
  /// Injects the operations into the Transform dialect. The operations must
  /// implement the TransformOpInterface and the implementation must be already
  /// available when the operation is injected.
  template <typename... OpTys>
  void registerTransformOps() {
    opInitializers.push_back([](TransformDialect *transformDialect) {
      transformDialect->addOperations<OpTys...>();

#ifndef NDEBUG
      (void)std::initializer_list<int>{
          (detail::checkImplementsTransformInterface<OpTys>(
               transformDialect->getContext()),
           0)...};
#endif // NDEBUG
    });
  }

  /// Declares that this Transform dialect extension depends on the dialect
  /// provided as template parameter. When the Transform dialect is loaded,
  /// dependent dialects will be loaded as well. This is intended for dialects
  /// that contain attributes and types used in creation and canonicalization of
  /// the injected operations.
  template <typename DialectTy>
  void declareDependentDialect() {
    dialectLoaders.push_back(
        [](MLIRContext *context) { context->loadDialect<DialectTy>(); });
  }

private:
  SmallVector<Initializer> opInitializers;
  SmallVector<DialectLoader> dialectLoaders;
};

} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMDIALECT_H
