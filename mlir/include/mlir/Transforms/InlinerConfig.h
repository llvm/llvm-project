//===- InlinerConfig.h - Config for the Inliner pass-------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares the config class used by the Inliner class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_INLINER_CONFIG_H
#define MLIR_TRANSFORMS_INLINER_CONFIG_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class OpPassManager;
class Operation;

class InlinerConfig {
public:
  using DefaultPipelineTy = std::function<void(OpPassManager &)>;
  using OpPipelinesTy = llvm::StringMap<OpPassManager>;

  InlinerConfig() = default;
  InlinerConfig(DefaultPipelineTy defaultPipeline,
                unsigned maxInliningIterations)
      : defaultPipeline(std::move(defaultPipeline)),
        maxInliningIterations(maxInliningIterations) {}

  const DefaultPipelineTy &getDefaultPipeline() const {
    return defaultPipeline;
  }
  const OpPipelinesTy &getOpPipelines() const { return opPipelines; }
  unsigned getMaxInliningIterations() const { return maxInliningIterations; }
  const InlinerInterface::CloneCallbackTy &getCloneCallback() const {
    return cloneCallback;
  }
  bool getCanHandleMultipleBlocks() const { return canHandleMultipleBlocks; }

  void setDefaultPipeline(DefaultPipelineTy pipeline) {
    defaultPipeline = std::move(pipeline);
  }
  void setOpPipelines(OpPipelinesTy pipelines) {
    opPipelines = std::move(pipelines);
  }
  void setMaxInliningIterations(unsigned max) { maxInliningIterations = max; }
  void setCloneCallback(InlinerInterface::CloneCallbackTy callback) {
    cloneCallback = std::move(callback);
  }
  void setCanHandleMultipleBlocks(bool value = true) {
    canHandleMultipleBlocks = value;
  }

private:
  /// An optional function that constructs an optimization pipeline for
  /// a given operation. This optimization pipeline is applied
  /// only to those callable operations that do not have dedicated
  /// optimization pipeline in opPipelines (based on the operation name).
  DefaultPipelineTy defaultPipeline;
  /// A map of operation names to pass pipelines to use when optimizing
  /// callable operations of these types. This provides a specialized pipeline
  /// instead of the one produced by defaultPipeline.
  OpPipelinesTy opPipelines;
  /// For SCC-based inlining algorithms, specifies maximum number of iterations
  /// when inlining within an SCC.
  unsigned maxInliningIterations{0};
  /// Callback for cloning operations during inlining
  InlinerInterface::CloneCallbackTy cloneCallback =
      [](OpBuilder &builder, Region *src, Block *inlineBlock,
         Block *postInsertBlock, IRMapping &mapper,
         bool shouldCloneInlinedRegion) {
        // Check to see if the region is being cloned, or moved inline. In
        // either case, move the new blocks after the 'insertBlock' to improve
        // IR readability.
        Region *insertRegion = inlineBlock->getParent();
        if (shouldCloneInlinedRegion)
          src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapper);
        else
          insertRegion->getBlocks().splice(postInsertBlock->getIterator(),
                                           src->getBlocks(), src->begin(),
                                           src->end());
      };
  /// Determine if the inliner can inline a function containing multiple
  /// blocks into a region that requires a single block. By default, it is
  /// not allowed. If it is true, cloneCallback should perform the extra
  /// transformation. see the example in
  /// mlir/test/lib/Transforms/TestInliningCallback.cpp
  bool canHandleMultipleBlocks{false};
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_INLINER_CONFIG_H
