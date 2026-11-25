//===- AttachSelectObject.cpp - Attach a select object handler ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuAttachSelectObjectPass` pass, attaching
// `#gpu.select_object` offloading handlers to GPU modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Regex.h"
#include <limits>

using namespace mlir;
using namespace mlir::gpu;

namespace mlir {
#define GEN_PASS_DEF_GPUATTACHSELECTOBJECTPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
class GpuAttachSelectObjectPass
    : public impl::GpuAttachSelectObjectPassBase<GpuAttachSelectObjectPass> {
public:
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

void GpuAttachSelectObjectPass::runOnOperation() {
  if (priority > std::numeric_limits<uint16_t>::max()) {
    getOperation()->emitError()
        << "Invalid priority specified: " << priority.getValue()
        << " (expected a value in the range [0, 65535])";
    return signalPassFailure();
  }
  // The default priority is the one of modules with no offloading handler.
  if (priority == SelectObjectAttr::kDefaultPriority)
    return;

  Builder builder(&getContext());
  llvm::Regex matcher(moduleMatcher);
  for (Region &region : getOperation()->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (GPUModuleOp module : block.getOps<GPUModuleOp>()) {
        // Check if the name of the module matches.
        if (!moduleMatcher.empty() && !matcher.match(module.getName()))
          continue;
        // Leave other offloading handlers, and handlers that already specify
        // a priority, unchanged.
        Attribute handler = module.getOffloadingHandlerAttr();
        auto selectObject = dyn_cast_or_null<SelectObjectAttr>(handler);
        if (handler && !selectObject)
          continue;
        if (selectObject &&
            selectObject.getPriority() != SelectObjectAttr::kDefaultPriority)
          continue;
        // A null target selects the first object.
        Attribute target =
            selectObject ? selectObject.getTarget() : Attribute();
        if (!target)
          target = builder.getI64IntegerAttr(0);
        module.setOffloadingHandlerAttr(
            SelectObjectAttr::get(&getContext(), target, priority));
      }
    }
  }
}
