//===- TransformInterpreterPassBase.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Base class with shared implementation for transform dialect interpreter
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERPASSBASE_H
#define MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERPASSBASE_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include <memory>

namespace mlir {
struct LogicalResult;
class MLIRContext;
class ModuleOp;
class Operation;
template <typename>
class OwningOpRef;
class Region;

namespace transform {
namespace detail {
/// Template-free implementation of TransformInterpreterPassBase::initialize.
LogicalResult
interpreterBaseInitializeImpl(MLIRContext *context, StringRef transformFileName,
                              std::shared_ptr<OwningOpRef<ModuleOp>> &module);

/// Template-free implementation of
/// TransformInterpreterPassBase::runOnOperation.
LogicalResult interpreterBaseRunOnOperationImpl(
    Operation *target, StringRef passName,
    const std::shared_ptr<OwningOpRef<ModuleOp>> &sharedTransformModule,
    const RaggedArray<MappedValue> &extraMappings,
    const TransformOptions &options,
    const Pass::Option<std::string> &transformFileName,
    const Pass::Option<std::string> &debugPayloadRootTag,
    const Pass::Option<std::string> &debugTransformRootTag,
    StringRef binaryName);
} // namespace detail

/// Base class for transform dialect interpreter passes that can consume and
/// dump transform dialect scripts in separate files. The pass is controlled by
/// three string options:
///
///   - transformFileName: if non-empty, the name of the file containing the
///     transform script. If empty, `debugTransformRootTag` is considered or the
///     pass root operation must contain a single top-level transform op that
///     will be interpreted.
///   - debugPayloadRootTag: if non-empty, the value of the attribute named
///     `kTransformDialectTagAttrName` indicating the single op that is
///     considered the payload root of the transform interpreter; otherwise, the
///     root operation of the pass is used.
///   - debugTransformRootTag: if non-empty, the value of the attribute named
///     `kTransformDialectTagAttrName` indicating the single top-level transform
///     op contained in the payload root to be used as the entry point by the
///     transform interpreter; mutually exclusive with `transformFileName`.
///
/// The pass runs the transform dialect interpreter as directed by the options.
/// It also provides the mechanism to dump reproducers into stderr
/// (-debug-only=transform-dialect-dump-repro) or into a temporary file
/// (-debug-only=transform-dialect-save-repro) that can be used with this
/// pass in a standalone mode.
///
/// Concrete passes must derive from this class instead of their generated base
/// class (or PassWrapper), and supply themselves and the generated base class
/// as template arguments. They are *not* expected to to implement `initialize`
/// or `runOnOperation`. They *are* expected to call the copy constructor of
/// this class in their copy constructors, short of which the file-based
/// transform dialect script injection facility will become nonoperational.
///
/// Concrete passes may implement the `runBeforeInterpreter` and
/// `runAfterInterpreter` to customize the behavior of the pass.
template <typename Concrete, template <typename> typename GeneratedBase>
class TransformInterpreterPassBase : public GeneratedBase<Concrete> {
public:
  explicit TransformInterpreterPassBase(
      const TransformOptions &options = TransformOptions())
      : options(options) {}

  TransformInterpreterPassBase(const TransformInterpreterPassBase &pass) {
    sharedTransformModule = pass.sharedTransformModule;
    options = pass.options;
  }

  static StringLiteral getBinaryName() { return "mlir-opt"; }

  LogicalResult initialize(MLIRContext *context) override {

#define REQUIRE_PASS_OPTION(NAME)                                              \
  static_assert(                                                               \
      std::is_same_v<                                                          \
          std::remove_reference_t<decltype(std::declval<Concrete &>().NAME)>,  \
          Pass::Option<std::string>>,                                          \
      "required " #NAME " string pass option is missing")

    REQUIRE_PASS_OPTION(transformFileName);
    REQUIRE_PASS_OPTION(debugPayloadRootTag);
    REQUIRE_PASS_OPTION(debugTransformRootTag);

#undef REQUIRE_PASS_OPTION

    StringRef transformFileName =
        static_cast<Concrete *>(this)->transformFileName;
    return detail::interpreterBaseInitializeImpl(context, transformFileName,
                                                 sharedTransformModule);
  }

  /// Hook for passes to run additional logic in the pass before the
  /// interpreter. If failure is returned, the pass fails and the interpreter is
  /// not run.
  LogicalResult runBeforeInterpreter(Operation *) { return success(); }

  /// Hook for passes to run additional logic in the pass after the interpreter.
  /// Only runs if everything succeeded before. If failure is returned, the pass
  /// fails.
  LogicalResult runAfterInterpreter(Operation *) { return success(); }

  void runOnOperation() override {
    auto *pass = static_cast<Concrete *>(this);
    Operation *op = pass->getOperation();
    StringRef binaryName = Concrete::getBinaryName();
    if (failed(pass->runBeforeInterpreter(op)) ||
        failed(detail::interpreterBaseRunOnOperationImpl(
            op, pass->getArgument(), sharedTransformModule,
            /*extraMappings=*/{}, options, pass->transformFileName,
            pass->debugPayloadRootTag, pass->debugTransformRootTag,
            binaryName)) ||
        failed(pass->runAfterInterpreter(op))) {
      return pass->signalPassFailure();
    }
  }

protected:
  /// Transform interpreter options.
  TransformOptions options;

  /// Returns a read-only reference to shared transform module.
  const std::shared_ptr<OwningOpRef<ModuleOp>> &
  getSharedTransformModule() const {
    return sharedTransformModule;
  }

private:
  /// The separate transform module to be used for transformations, shared
  /// across multiple instances of the pass if it is applied in parallel to
  /// avoid potentially expensive cloning. MUST NOT be modified after the pass
  /// has been initialized.
  std::shared_ptr<OwningOpRef<ModuleOp>> sharedTransformModule = nullptr;
};

} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINTERPRETERPASSBASE_H
