//===- OpenACCUtilsGPU.cpp - OpenACC GPU Utilities ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for OpenACC that depend on the GPU
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsGPU.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace acc {

std::optional<gpu::GPUModuleOp> getOrCreateGPUModule(ModuleOp mod, bool create,
                                                     llvm::StringRef name) {
  // Use default name if provided name is empty
  llvm::StringRef moduleName =
      name.empty() ? llvm::StringRef(kDefaultGPUModuleName) : name;

  // Look for existing GPU module with the specified name
  SymbolTable symTab(mod);
  if (auto gpuMod = symTab.lookup<gpu::GPUModuleOp>(moduleName))
    return gpuMod;

  if (!create)
    return std::nullopt;

  // Create a new GPU module
  auto *ctx = mod.getContext();
  mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
               UnitAttr::get(ctx));

  OpBuilder builder(ctx);
  auto gpuMod = gpu::GPUModuleOp::create(builder, mod.getLoc(), moduleName);
  Block::iterator insertPt(mod.getBodyRegion().front().end());
  symTab.insert(gpuMod, insertPt);
  return gpuMod;
}

static Value getGPUSizeFromLaunch(gpu::LaunchOp launch,
                                  gpu::Processor processor) {
  gpu::KernelDim3 gridSize = launch.getGridSize();
  gpu::KernelDim3 blockSize = launch.getBlockSize();
  switch (processor) {
  case gpu::Processor::ThreadX:
    return blockSize.x;
  case gpu::Processor::ThreadY:
    return blockSize.y;
  case gpu::Processor::ThreadZ:
    return blockSize.z;
  case gpu::Processor::BlockX:
    return gridSize.x;
  case gpu::Processor::BlockY:
    return gridSize.y;
  case gpu::Processor::BlockZ:
    return gridSize.z;
  default:
    return {};
  }
}

Value getGPUSize(gpu::Processor processor, gpu::LaunchOp launch,
                 const llvm::DenseMap<gpu::Processor, Value> &dimensionOps) {
  if (launch)
    return getGPUSizeFromLaunch(launch, processor);
  assert(!dimensionOps.empty() && "dimension map is empty");
  return dimensionOps.lookup(processor);
}

static Value getGPUThreadIdFromLaunch(gpu::LaunchOp launch,
                                      gpu::Processor processor) {
  gpu::KernelDim3 blockIds = launch.getBlockIds();
  gpu::KernelDim3 threadIds = launch.getThreadIds();
  switch (processor) {
  case gpu::Processor::ThreadX:
    return threadIds.x;
  case gpu::Processor::ThreadY:
    return threadIds.y;
  case gpu::Processor::ThreadZ:
    return threadIds.z;
  case gpu::Processor::BlockX:
    return blockIds.x;
  case gpu::Processor::BlockY:
    return blockIds.y;
  case gpu::Processor::BlockZ:
    return blockIds.z;
  default:
    return {};
  }
}

Value getGPUThreadId(gpu::Processor processor, gpu::LaunchOp launch,
                     const llvm::DenseMap<gpu::Processor, Value> &indexOps) {
  if (launch)
    return getGPUThreadIdFromLaunch(launch, processor);
  assert(!indexOps.empty() && "index map is empty");
  return indexOps.lookup(processor);
}

} // namespace acc
} // namespace mlir
