//===- InferEffects.cpp - Infer memory effects for named symbols ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mlir {
namespace transform {
#define GEN_PASS_DEF_INFEREFFECTSPASS
#include "mlir/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
} // namespace mlir

static LogicalResult inferSideEffectAnnotations(Operation *op) {
  if (!isa<transform::TransformOpInterface>(op))
    return success();

  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func || func.isExternal())
    return success();

  if (!func.getFunctionBody().hasOneBlock()) {
    return op->emitError()
           << "only single-block operations are currently supported";
  }

  // Note that there can't be an inclusion of an unannotated symbol because it
  // wouldn't have passed the verifier, so recursion isn't necessary here.
  llvm::SmallDenseSet<unsigned> consumedArguments;
  transform::getConsumedBlockArguments(func.getFunctionBody().front(),
                                       consumedArguments);

  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    func.setArgAttr(i,
                    consumedArguments.contains(i)
                        ? transform::TransformDialect::kArgConsumedAttrName
                        : transform::TransformDialect::kArgReadOnlyAttrName,
                    UnitAttr::get(op->getContext()));
  }
  return success();
}

namespace {
class InferEffectsPass
    : public transform::impl::InferEffectsPassBase<InferEffectsPass> {
public:
  void runOnOperation() override {
    WalkResult result = getOperation()->walk([](Operation *op) {
      return failed(inferSideEffectAnnotations(op)) ? WalkResult::interrupt()
                                                    : WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
