//=== AffineTransformOps.cpp - Implementation of Affine transformation ops ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

using namespace mlir;

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// AffineGetParentForOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::AffineGetParentForOp::apply(transform::TransformResults &results,
                                       transform::TransformState &state) {
  SetVector<Operation *> parents;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    AffineForOp loop;
    Operation *current = target;
    for (unsigned i = 0, e = getNumLoops(); i < e; ++i) {
      loop = current->getParentOfType<AffineForOp>();
      if (!loop) {
        DiagnosedSilenceableFailure diag = emitSilenceableError()
                                           << "could not find an '"
                                           << AffineForOp::getOperationName()
                                           << "' parent";
        diag.attachNote(target->getLoc()) << "target op";
        results.set(getResult().cast<OpResult>(), {});
        return diag;
      }
      current = loop;
    }
    parents.insert(loop);
  }
  results.set(getResult().cast<OpResult>(), parents.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LoopUnrollOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::AffineLoopUnrollOp::applyToOne(AffineForOp target,
                                          SmallVector<Operation *> &results,
                                          transform::TransformState &state) {
  if (failed(loopUnrollByFactor(target, getFactor()))) {
    Diagnostic diag(target->getLoc(), DiagnosticSeverity::Note);
    diag << "op failed to unroll";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class AffineTransformDialectExtension
    : public transform::TransformDialectExtension<
          AffineTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<AffineDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.cpp.inc"

void mlir::affine::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<AffineTransformDialectExtension>();
}
