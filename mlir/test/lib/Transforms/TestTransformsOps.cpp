//===- TestTransformsOps.cpp - Test Transforms ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines transform dialect operations for testing MLIR
// transformations
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#define GET_OP_CLASSES
#include "TestTransformsOps.h.inc"

using namespace mlir;
using namespace mlir::transform;

#define GET_OP_CLASSES
#include "TestTransformsOps.cpp.inc"

DiagnosedSilenceableFailure
transform::TestMoveOperandDeps::apply(TransformRewriter &rewriter,
                                      TransformResults &TransformResults,
                                      TransformState &state) {
  Operation *op = *state.getPayloadOps(getOp()).begin();
  Operation *moveBefore = *state.getPayloadOps(getInsertionPoint()).begin();
  if (failed(moveOperationDependencies(rewriter, op, moveBefore))) {
    auto listener = cast<ErrorCheckingTrackingListener>(rewriter.getListener());
    std::string errorMsg = listener->getLatestMatchFailureMessage();
    (void)emitRemark(errorMsg);
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::TestMoveValueDefns::apply(TransformRewriter &rewriter,
                                     TransformResults &TransformResults,
                                     TransformState &state) {
  SmallVector<Value> values;
  for (auto tdValue : getValues()) {
    values.push_back(*state.getPayloadValues(tdValue).begin());
  }
  Operation *moveBefore = *state.getPayloadOps(getInsertionPoint()).begin();
  if (failed(moveValueDefinitions(rewriter, values, moveBefore))) {
    auto listener = cast<ErrorCheckingTrackingListener>(rewriter.getListener());
    std::string errorMsg = listener->getLatestMatchFailureMessage();
    (void)emitRemark(errorMsg);
  }
  return DiagnosedSilenceableFailure::success();
}

namespace {

class TestTransformsDialectExtension
    : public transform::TransformDialectExtension<
          TestTransformsDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTransformsDialectExtension)

  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "TestTransformsOps.cpp.inc"
        >();
  }
};
} // namespace

namespace test {
void registerTestTransformsTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<TestTransformsDialectExtension>();
}
} // namespace test
