//===- OpenACCSupport.h - OpenACC Support Interface -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpenACCSupport analysis interface, which provides
// extensible support for OpenACC passes. Custom implementations
// can be registered to provide pipeline and dialect-specific information
// that cannot be adequately expressed through type or operation interfaces
// alone.
//
// Usage Pattern:
// ==============
//
// A pass that needs this functionality should call
// getAnalysis<OpenACCSupport>(), which will provide either:
// - A cached version if previously initialized, OR
// - A default implementation if not previously initialized
//
// This analysis is never invalidated (isInvalidated returns false), so it only
// needs to be initialized once and will persist throughout the pass pipeline.
//
// Registering a Custom Implementation:
// =====================================
//
// If a custom implementation is needed, create a pass that runs BEFORE the pass
// that needs the analysis. In this setup pass, use
// getAnalysis<OpenACCSupport>() followed by setImplementation() to register
// your custom implementation. The custom implementation will need to provide
// implementation for all methods defined in the `OpenACCSupportTraits::Concept`
// class.
//
// Example:
//   void MySetupPass::runOnOperation() {
//     OpenACCSupport &support = getAnalysis<OpenACCSupport>();
//     support.setImplementation(MyCustomImpl());
//   }
//
//   void MyAnalysisConsumerPass::runOnOperation() {
//     OpenACCSupport &support = getAnalysis<OpenACCSupport>();
//     std::string name = support.getVariableName(someValue);
//     // ... use the analysis results
//   }
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_ANALYSIS_OPENACCSUPPORT_H
#define MLIR_DIALECT_OPENACC_ANALYSIS_OPENACCSUPPORT_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsGPU.h"
#include "mlir/IR/Remarks.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>
#include <string>

namespace mlir {
namespace acc {

namespace detail {
/// This class contains internal trait classes used by OpenACCSupport.
/// It follows the Concept-Model pattern used throughout MLIR (e.g., in
/// AliasAnalysis and interface definitions).
struct OpenACCSupportTraits {
  class Concept {
  public:
    virtual ~Concept() = default;

    /// Get the variable name for a given MLIR value.
    virtual std::string getVariableName(Value v) = 0;

    /// Get the recipe name for a given kind, type and value.
    virtual std::string getRecipeName(RecipeKind kind, Type type,
                                      Value var) = 0;

    // Used to report a case that is not supported by the implementation.
    virtual InFlightDiagnostic emitNYI(Location loc, const Twine &message) = 0;

    // Used to emit an OpenACC remark. The category is optional and is used to
    // either capture the pass name or pipeline phase when the remark is
    // emitted. When not provided, in the default implementation, the category
    // is "openacc".
    virtual remark::detail::InFlightRemark
    emitRemark(Operation *op, std::function<std::string()> messageFn,
               llvm::StringRef category) = 0;

    /// Check if a symbol use is valid for use in an OpenACC region.
    virtual bool isValidSymbolUse(Operation *user, SymbolRefAttr symbol,
                                  Operation **definingOpPtr) = 0;

    /// Check if a value use is legal in an OpenACC region.
    virtual bool isValidValueUse(Value v, mlir::Region &region) = 0;

    /// Get or optionally create a GPU module in the given module.
    virtual std::optional<gpu::GPUModuleOp>
    getOrCreateGPUModule(ModuleOp mod, bool create, llvm::StringRef name) = 0;
  };

  /// SFINAE helpers to detect if implementation has optional methods
  template <typename ImplT, typename... Args>
  using isValidSymbolUse_t =
      decltype(std::declval<ImplT>().isValidSymbolUse(std::declval<Args>()...));

  template <typename ImplT>
  using has_isValidSymbolUse =
      llvm::is_detected<isValidSymbolUse_t, ImplT, Operation *, SymbolRefAttr,
                        Operation **>;

  template <typename ImplT, typename... Args>

  using isValidValueUse_t =
      decltype(std::declval<ImplT>().isValidValueUse(std::declval<Args>()...));

  template <typename ImplT>
  using has_isValidValueUse =
      llvm::is_detected<isValidValueUse_t, ImplT, Value, Region &>;

  template <typename ImplT, typename... Args>
  using emitRemark_t =
      decltype(std::declval<ImplT>().emitRemark(std::declval<Args>()...));

  template <typename ImplT>
  using has_emitRemark =
      llvm::is_detected<emitRemark_t, ImplT, Operation *,
                        std::function<std::string()>, llvm::StringRef>;

  template <typename ImplT, typename... Args>
  using getOrCreateGPUModule_t =
      decltype(std::declval<ImplT>().getOrCreateGPUModule(
          std::declval<Args>()...));

  template <typename ImplT>
  using has_getOrCreateGPUModule =
      llvm::is_detected<getOrCreateGPUModule_t, ImplT, ModuleOp, bool,
                        llvm::StringRef>;

  /// This class wraps a concrete OpenACCSupport implementation and forwards
  /// interface calls to it. This provides type erasure, allowing different
  /// implementation types to be used interchangeably without inheritance.
  /// Methods can be optionally implemented; if not present, default behavior
  /// is used.
  template <typename ImplT>
  class Model final : public Concept {
  public:
    explicit Model(ImplT &&impl) : impl(std::forward<ImplT>(impl)) {}
    ~Model() override = default;

    std::string getVariableName(Value v) final {
      return impl.getVariableName(v);
    }

    std::string getRecipeName(RecipeKind kind, Type type, Value var) final {
      return impl.getRecipeName(kind, type, var);
    }

    InFlightDiagnostic emitNYI(Location loc, const Twine &message) final {
      return impl.emitNYI(loc, message);
    }

    remark::detail::InFlightRemark
    emitRemark(Operation *op, std::function<std::string()> messageFn,
               llvm::StringRef category) final {
      if constexpr (has_emitRemark<ImplT>::value)
        return impl.emitRemark(op, std::move(messageFn), category);
      else
        return acc::emitRemark(op, messageFn(), category);
    }

    bool isValidSymbolUse(Operation *user, SymbolRefAttr symbol,
                          Operation **definingOpPtr) final {
      if constexpr (has_isValidSymbolUse<ImplT>::value)
        return impl.isValidSymbolUse(user, symbol, definingOpPtr);
      else
        return acc::isValidSymbolUse(user, symbol, definingOpPtr);
    }

    bool isValidValueUse(Value v, Region &region) final {
      if constexpr (has_isValidValueUse<ImplT>::value)
        return impl.isValidValueUse(v, region);
      else
        return acc::isValidValueUse(v, region);
    }

    std::optional<gpu::GPUModuleOp>
    getOrCreateGPUModule(ModuleOp mod, bool create,
                         llvm::StringRef name) final {
      if constexpr (has_getOrCreateGPUModule<ImplT>::value)
        return impl.getOrCreateGPUModule(mod, create, name);
      else
        return acc::getOrCreateGPUModule(mod, create, name);
    }

  private:
    ImplT impl;
  };
};
} // namespace detail

//===----------------------------------------------------------------------===//
// OpenACCSupport
//===----------------------------------------------------------------------===//

class OpenACCSupport {
  using Concept = detail::OpenACCSupportTraits::Concept;
  template <typename ImplT>
  using Model = detail::OpenACCSupportTraits::Model<ImplT>;

public:
  OpenACCSupport() = default;
  OpenACCSupport(Operation *op) {}

  /// Register a custom OpenACCSupport implementation. Only one implementation
  /// can be registered at a time; calling this replaces any existing
  /// implementation.
  template <typename AnalysisT>
  void setImplementation(AnalysisT &&analysis) {
    impl =
        std::make_unique<Model<AnalysisT>>(std::forward<AnalysisT>(analysis));
  }

  /// Get the variable name for a given value.
  ///
  /// \param v The MLIR value to get the variable name for.
  /// \return The variable name, or an empty string if unavailable.
  std::string getVariableName(Value v);

  /// Get the recipe name for a given type and value.
  ///
  /// \param kind The kind of recipe to get the name for.
  /// \param type The type to get the recipe name for. Can be null if the
  ///        var is provided instead.
  /// \param var The MLIR value to get the recipe name for. Can be null if
  ///        the type is provided instead.
  /// \return The recipe name, or an empty string if not available.
  std::string getRecipeName(RecipeKind kind, Type type, Value var);

  /// Report a case that is not yet supported by the implementation.
  ///
  /// \param loc The location to report the unsupported case at.
  /// \param message The message to report.
  /// \return An in-flight diagnostic object that can be used to report the
  ///         unsupported case.
  InFlightDiagnostic emitNYI(Location loc, const Twine &message);

  /// Emit an OpenACC remark with lazy message generation.
  ///
  /// The messageFn is only invoked if remarks are enabled for the given
  /// operation, allowing callers to avoid constructing expensive messages
  /// when remarks are disabled.
  ///
  /// \param op The operation to emit the remark for.
  /// \param messageFn A callable that returns the remark message.
  /// \param category Optional category for the remark. Defaults to "openacc".
  /// \return An in-flight remark object that can be used to append
  ///         additional information to the remark.
  remark::detail::InFlightRemark
  emitRemark(Operation *op, std::function<std::string()> messageFn,
             llvm::StringRef category = "openacc");

  /// Emit an OpenACC remark.
  ///
  /// \param op The operation to emit the remark for.
  /// \param message The remark message.
  /// \param category Optional category for the remark. Defaults to "openacc".
  /// \return An in-flight remark object that can be used to append
  ///         additional information to the remark.
  remark::detail::InFlightRemark
  emitRemark(Operation *op, const Twine &message,
             llvm::StringRef category = "openacc") {
    return emitRemark(op, std::function<std::string()>([msg = message.str()]() {
                        return msg;
                      }),
                      category);
  }

  /// Check if a symbol use is valid for use in an OpenACC region.
  ///
  /// \param user The operation using the symbol.
  /// \param symbol The symbol reference being used.
  /// \param definingOpPtr Optional output parameter to receive the defining op.
  /// \return true if the symbol use is valid, false otherwise.
  bool isValidSymbolUse(Operation *user, SymbolRefAttr symbol,
                        Operation **definingOpPtr = nullptr);

  /// Check if a value use is legal in an OpenACC region.
  ///
  /// \param v The MLIR value to check for legality.
  /// \param region The MLIR region in which the legality is checked.
  bool isValidValueUse(Value v, Region &region);

  /// Get or optionally create a GPU module in the given module.
  ///
  /// \param mod The module to search or create the GPU module in.
  /// \param create If true (default), create the GPU module if it doesn't
  /// exist.
  /// \param name The name for the GPU module. If empty, implementation uses its
  ///        default name.
  /// \return The GPU module if found or created, std::nullopt otherwise.
  std::optional<gpu::GPUModuleOp>
  getOrCreateGPUModule(ModuleOp mod, bool create = true,
                       llvm::StringRef name = "");

  /// Signal that this analysis should always be preserved so that
  /// underlying implementation registration is not lost.
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return false;
  }

private:
  /// The registered custom implementation (if any).
  std::unique_ptr<Concept> impl;
};

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_ANALYSIS_OPENACCSUPPORT_H
