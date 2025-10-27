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

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
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
  };

  /// This class wraps a concrete OpenACCSupport implementation and forwards
  /// interface calls to it. This provides type erasure, allowing different
  /// implementation types to be used interchangeably without inheritance.
  template <typename ImplT>
  class Model final : public Concept {
  public:
    explicit Model(ImplT &&impl) : impl(std::forward<ImplT>(impl)) {}
    ~Model() override = default;

    std::string getVariableName(Value v) final {
      return impl.getVariableName(v);
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
