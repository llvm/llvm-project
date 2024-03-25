//===- ConvertGPUToVulkanPass.h - GPU to Vulkan conversion pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file declares a pass to convert GPU dialect ops to to Vulkan runtime
// calls.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOVULKAN_CONVERTGPUTOVULKANPASS_H
#define MLIR_CONVERSION_GPUTOVULKAN_CONVERTGPUTOVULKANPASS_H

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T>
class OperationPass;
class Pass;

#define GEN_PASS_DECL_CONVERTVULKANLAUNCHFUNCTOVULKANCALLSPASS
#define GEN_PASS_DECL_CONVERTGPULAUNCHFUNCTOVULKANLAUNCHFUNC
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertGpuLaunchFuncToVulkanLaunchFuncPass();

} // namespace mlir
#endif // MLIR_CONVERSION_GPUTOVULKAN_CONVERTGPUTOVULKANPASS_H
