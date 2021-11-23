//===- ComprehensiveBufferize.h - Linalg bufferization pass -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"

namespace mlir {

class ModuleOp;

namespace linalg {
namespace comprehensive_bufferize {

// TODO: from some HW description.
static constexpr int64_t kBufferAlignments = 128;

/// Return default allocation callbacks.
std::unique_ptr<AllocationCallbacks> defaultAllocationCallbacks();

/// Bufferize one particular op.
LogicalResult bufferizeOp(Operation *op, BufferizationState &state);

/// Register external models implemented for the `BufferizableOpInterface`.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

/// Options for ComprehensiveBufferize.
struct BufferizationOptions {
  BufferizationOptions();

  // BufferizationOptions cannot be copied.
  BufferizationOptions(const BufferizationOptions &other) = delete;

  /// Register a "post analysis" step. Such steps are executed after the
  /// analysis, but before bufferization.
  template <typename Step, typename... Args>
  void addPostAnalysisStep(Args... args) {
    postAnalysisSteps.emplace_back(
        std::make_unique<Step>(std::forward<Args>(args)...));
  }

  /// Helper functions for allocation, deallocation, memory copying.
  std::unique_ptr<AllocationCallbacks> allocationFns;

  /// Specifies whether returning newly allocated memrefs should be allowed.
  /// Otherwise, a pass failure is triggered.
  bool allowReturnMemref = false;

  /// Seed for the analysis fuzzer. If set to `0`, the fuzzer is deactivated.
  /// Should be used only with `testAnalysisOnly = true`.
  unsigned analysisFuzzerSeed = 0;

  /// If set to `true`, does not modify the IR apart from adding attributes (for
  /// checking the results of the analysis) and post analysis steps.
  bool testAnalysisOnly = false;

  /// Registered post analysis steps.
  std::vector<std::unique_ptr<PostAnalysisStep>> postAnalysisSteps;
};

LogicalResult runComprehensiveBufferize(ModuleOp moduleOp,
                                        const BufferizationOptions &options);

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_COMPREHENSIVE_BUFFERIZE_H
