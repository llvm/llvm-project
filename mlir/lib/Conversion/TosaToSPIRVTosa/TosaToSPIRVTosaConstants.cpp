//===- TosaToSPIRVTosaConstants.cpp - TOSA graph constants ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements preprocessing that marks TOSA constants that should be
// lowered to SPIR-V Graph constants.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToSPIRVTosa/TosaToSPIRVTosa.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_TOSATOSPIRVTOSAMARKGRAPHCONSTANTS
#include "mlir/Conversion/Passes.h.inc"

namespace tosa {
namespace {

constexpr uint32_t maxInlineConstElements = 16;
constexpr uint32_t maxInlineConstShapeElements = 32;

std::optional<ElementsAttr> getConstantValues(Operation *op) {
  if (auto constOp = dyn_cast<tosa::ConstOp>(op))
    return constOp.getValuesAttr();
  if (auto constShapeOp = dyn_cast<tosa::ConstShapeOp>(op))
    return constShapeOp.getValuesAttr();
  return std::nullopt;
}

bool shouldMarkGraphConstant(Operation *op) {
  if (op->use_empty())
    return false;

  std::optional<ElementsAttr> values = getConstantValues(op);
  if (!values)
    return false;

  uint32_t maxInlineElements = isa<tosa::ConstOp>(op)
                                   ? maxInlineConstElements
                                   : maxInlineConstShapeElements;
  return values->size() > maxInlineElements;
}

void setGraphConstantId(Operation *op, uint32_t id) {
  auto i32Type = IntegerType::get(op->getContext(), 32);
  op->setAttr(graphARMGraphConstantIdAttrName, IntegerAttr::get(i32Type, id));
}

struct TosaToSPIRVTosaMarkGraphConstants final
    : impl::TosaToSPIRVTosaMarkGraphConstantsBase<
          TosaToSPIRVTosaMarkGraphConstants> {
  void runOnOperation() override {
    uint32_t nextConstantId = 0;
    WalkResult result =
        getOperation().walk([&](Operation *op) {
          if (!isa<tosa::ConstOp, tosa::ConstShapeOp>(op))
            return WalkResult::advance();

          if (op->hasAttr(graphARMGraphConstantIdAttrName)) {
            op->emitOpError()
                << "already has `" << graphARMGraphConstantIdAttrName
                << "`; this pass assigns graph constant IDs automatically and "
                   "does not support pre-marked constants";
            return WalkResult::interrupt();
          }

          if (shouldMarkGraphConstant(op))
            setGraphConstantId(op, nextConstantId++);
          return WalkResult::advance();
        });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createTosaToSPIRVTosaMarkGraphConstants() {
  return std::make_unique<TosaToSPIRVTosaMarkGraphConstants>();
}

} // namespace tosa
} // namespace mlir
