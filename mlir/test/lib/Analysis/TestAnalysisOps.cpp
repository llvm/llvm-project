//===- TestAnalysisOps.cpp - Test Transforms ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines transform dialect operations for testing MLIR
// analyses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "TestAnalysisOps.h.inc"

using namespace mlir;
using namespace transform;

#define GET_OP_CLASSES
#include "TestAnalysisOps.cpp.inc"

/// Create a function with the same signature as the parent function of `op`
/// with name being the function name and a `suffix`.
static LogicalResult
createBackwardSliceFunction(Operation *op, StringRef suffix,
                            const BackwardSliceOptions &options) {
  func::FuncOp parentFuncOp = op->getParentOfType<func::FuncOp>();
  if (!parentFuncOp)
    return failure();
  OpBuilder builder(parentFuncOp);
  Location loc = op->getLoc();
  std::string clonedFuncOpName = parentFuncOp.getName().str() + suffix.str();
  func::FuncOp clonedFuncOp = func::FuncOp::create(
      builder, loc, clonedFuncOpName, parentFuncOp.getFunctionType());
  IRMapping mapper;
  builder.setInsertionPointToEnd(clonedFuncOp.addEntryBlock());
  for (const auto &arg : enumerate(parentFuncOp.getArguments()))
    mapper.map(arg.value(), clonedFuncOp.getArgument(arg.index()));
  SetVector<Operation *> slice;
  LogicalResult result = getBackwardSlice(op, &slice, options);
  assert(result.succeeded() && "expected a backward slice");
  (void)result;
  for (Operation *slicedOp : slice)
    builder.clone(*slicedOp, mapper);
  func::ReturnOp::create(builder, loc);
  return success();
}

DiagnosedSilenceableFailure
transform::TestGetBackwardSlice::apply(TransformRewriter &rewriter,
                                       TransformResults &transformResults,
                                       TransformState &state) {
  Operation *op = *state.getPayloadOps(getOp()).begin();
  StringRef suffix = "__backward_slice__";
  BackwardSliceOptions options;
  options.omitBlockArguments = true;
  // TODO: Make this default.
  options.omitUsesFromAbove = false;
  options.inclusive = true;
  if (failed(createBackwardSliceFunction(op, suffix, options)))
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Extension
//===----------------------------------------------------------------------===//
namespace {

class TestAnalysisDialectExtension
    : public transform::TransformDialectExtension<
          TestAnalysisDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAnalysisDialectExtension)
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "TestAnalysisOps.cpp.inc"
        >();
  }
};
} // namespace

namespace test {
void registerTestAnalysisTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TestAnalysisDialectExtension>();
}
} // namespace test
