//===- BufferizationTransformOps.h - Bufferization transform ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// OneShotBufferizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::OneShotBufferizeOp::apply(TransformResults &transformResults,
                                     TransformState &state) {
  OneShotBufferizationOptions options;
  options.allowReturnAllocs = getAllowReturnAllocs();
  options.allowUnknownOps = getAllowUnknownOps();
  options.bufferizeFunctionBoundaries = getBufferizeFunctionBoundaries();
  options.createDeallocs = getCreateDeallocs();
  options.testAnalysisOnly = getTestAnalysisOnly();
  options.printConflicts = getPrintConflicts();

  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  for (Operation *target : payloadOps) {
    auto moduleOp = dyn_cast<ModuleOp>(target);
    if (getTargetIsModule() && !moduleOp)
      return emitSilenceableError() << "expected ModuleOp target";
    if (options.bufferizeFunctionBoundaries) {
      if (!moduleOp)
        return emitSilenceableError() << "expected ModuleOp target";
      if (failed(bufferization::runOneShotModuleBufferize(moduleOp, options)))
        return emitSilenceableError() << "bufferization failed";
    } else {
      if (failed(bufferization::runOneShotBufferize(target, options)))
        return emitSilenceableError() << "bufferization failed";
    }
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::OneShotBufferizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getTarget(),
                       TransformMappingResource::get());

  // Handles that are not modules are not longer usable.
  if (!getTargetIsModule())
    effects.emplace_back(MemoryEffects::Free::get(), getTarget(),
                         TransformMappingResource::get());
}
//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the additional
/// ops are using PDL types for operands and results.
class BufferizationTransformDialectExtension
    : public transform::TransformDialectExtension<
          BufferizationTransformDialectExtension> {
public:
  BufferizationTransformDialectExtension() {
    declareDependentDialect<bufferization::BufferizationDialect>();
    declareDependentDialect<pdl::PDLDialect>();
    declareDependentDialect<memref::MemRefDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.cpp.inc"

void mlir::bufferization::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<BufferizationTransformDialectExtension>();
}
