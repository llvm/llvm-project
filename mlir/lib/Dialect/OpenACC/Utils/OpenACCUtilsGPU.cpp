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

} // namespace acc
} // namespace mlir
