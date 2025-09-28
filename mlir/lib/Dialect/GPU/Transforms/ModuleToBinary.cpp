//===- ModuleToBinary.cpp - Transforms GPU modules to GPU binaries ----------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the `GpuModuleToBinaryPass` pass, transforming GPU
// modules into GPU binaries.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::gpu;

namespace mlir {
#define GEN_PASS_DEF_GPUMODULETOBINARYPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
class GpuModuleToBinaryPass
    : public impl::GpuModuleToBinaryPassBase<GpuModuleToBinaryPass> {
public:
  using Base::Base;
  void runOnOperation() final;
};
} // namespace

void GpuModuleToBinaryPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto targetFormat =
      llvm::StringSwitch<std::optional<CompilationTarget>>(compilationTarget)
          .Cases("offloading", "llvm", CompilationTarget::Offload)
          .Cases("assembly", "isa", CompilationTarget::Assembly)
          .Cases("binary", "bin", CompilationTarget::Binary)
          .Cases("fatbinary", "fatbin", CompilationTarget::Fatbin)
          .Default(std::nullopt);
  if (!targetFormat)
    getOperation()->emitError() << "Invalid format specified.";

  // Lazy symbol table builder callback.
  std::optional<SymbolTable> parentTable;
  auto lazyTableBuilder = [&]() -> SymbolTable * {
    // Build the table if it has not been built.
    if (!parentTable) {
      Operation *table = SymbolTable::getNearestSymbolTable(getOperation());
      // It's up to the target attribute to determine if failing to find a
      // symbol table is an error.
      if (!table)
        return nullptr;
      parentTable = SymbolTable(table);
    }
    return &parentTable.value();
  };
  SmallVector<Attribute> librariesToLink;
  for (const std::string &path : linkFiles)
    librariesToLink.push_back(StringAttr::get(&getContext(), path));
  TargetOptions targetOptions(toolkitPath, librariesToLink, cmdOptions,
                              elfSection, *targetFormat, lazyTableBuilder);
  if (failed(transformGpuModulesToBinaries(
          getOperation(), OffloadingLLVMTranslationAttrInterface(nullptr),
          targetOptions)))
    return signalPassFailure();
}

namespace {
LogicalResult moduleSerializer(GPUModuleOp op,
                               OffloadingLLVMTranslationAttrInterface handler,
                               const TargetOptions &targetOptions) {
  OpBuilder builder(op->getContext());
  SmallVector<Attribute> objects;
  // Fail if there are no target attributes
  if (!op.getTargetsAttr())
    return op.emitError("the module has no target attributes");
  // Serialize all targets.
  for (auto targetAttr : op.getTargetsAttr()) {
    assert(targetAttr && "Target attribute cannot be null.");
    auto target = dyn_cast<gpu::TargetAttrInterface>(targetAttr);
    assert(target &&
           "Target attribute doesn't implements `TargetAttrInterface`.");
    std::optional<SmallVector<char, 0>> serializedModule =
        target.serializeToObject(op, targetOptions);
    if (!serializedModule) {
      op.emitError("An error happened while serializing the module.");
      return failure();
    }

    Attribute object =
        target.createObject(op, *serializedModule, targetOptions);
    if (!object) {
      op.emitError("An error happened while creating the object.");
      return failure();
    }
    objects.push_back(object);
  }
  if (auto moduleHandler =
          dyn_cast_or_null<OffloadingLLVMTranslationAttrInterface>(
              op.getOffloadingHandlerAttr());
      !handler && moduleHandler)
    handler = moduleHandler;
  builder.setInsertionPointAfter(op);
  gpu::BinaryOp::create(builder, op.getLoc(), op.getName(), handler,
                        builder.getArrayAttr(objects));
  op->erase();
  return success();
}
} // namespace

LogicalResult mlir::gpu::transformGpuModulesToBinaries(
    Operation *op, OffloadingLLVMTranslationAttrInterface handler,
    const gpu::TargetOptions &targetOptions) {
  for (Region &region : op->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module :
           llvm::make_early_inc_range(block.getOps<GPUModuleOp>()))
        if (failed(moduleSerializer(module, handler, targetOptions)))
          return failure();
  return success();
}
