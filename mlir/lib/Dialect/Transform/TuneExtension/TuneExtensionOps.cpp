//===- TuneExtensionOps.cpp - Tune extension for the Transform dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Transform/TuneExtension/TuneExtensionOps.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/TuneExtension/TuneExtensionOps.cpp.inc"

#define DEBUG_TYPE "transform-tune"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

//===----------------------------------------------------------------------===//
// KnobOp
//===----------------------------------------------------------------------===//

void transform::tune::KnobOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
transform::tune::KnobOp::apply(transform::TransformRewriter &rewriter,
                               transform::TransformResults &results,
                               transform::TransformState &state) {
  if (getSelected()) {
    results.setParams(llvm::cast<OpResult>(getResult()), *getSelected());
    return DiagnosedSilenceableFailure::success();
  }

  return emitDefiniteFailure()
         << "non-deterministic choice " << getName()
         << " is only resolved through providing a `selected` attr";
}

LogicalResult transform::tune::KnobOp::verify() {
  if (auto selected = getSelected()) {
    if (auto optionsArray = dyn_cast<ArrayAttr>(getOptions())) {
      if (!llvm::is_contained(optionsArray, selected))
        return emitOpError("provided `selected` attribute is not an element of "
                           "`options` array of attributes");
    } else
      LLVM_DEBUG(DBGS() << "cannot verify `selected` attribute " << selected
                        << " is an element of `options` attribute "
                        << getOptions());
  }

  return success();
}
