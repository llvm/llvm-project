//===- GPUCommonPass.h - MLIR GPU runtime support -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_
#define MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_

#include "mlir/Support/LLVM.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mlir {

class Location;
class ModuleOp;

template <typename T>
class OperationPass;

/// Creates a pass to convert a gpu.launch_func operation into a sequence of
/// GPU runtime calls.
///
/// This pass does not generate code to call GPU runtime APIs directly but
/// instead uses a small wrapper library that exports a stable and conveniently
/// typed ABI on top of GPU runtimes such as CUDA or ROCm (HIP).
std::unique_ptr<OperationPass<ModuleOp>>
createConvertGpuLaunchFuncToGpuRuntimeCallsPass();

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_
