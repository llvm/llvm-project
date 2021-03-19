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
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {

class LLVMTypeConverter;
class Location;
struct LogicalResult;
class ModuleOp;
class Operation;
class OwningRewritePatternList;

template <typename T>
class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

using OwnedBlob = std::unique_ptr<std::vector<char>>;
using BlobGenerator =
    std::function<OwnedBlob(const std::string &, Location, StringRef)>;
using LoweringCallback = std::function<std::unique_ptr<llvm::Module>(
    Operation *, llvm::LLVMContext &, StringRef)>;

/// Creates a pass to convert a gpu.launch_func operation into a sequence of
/// GPU runtime calls.
///
/// This pass does not generate code to call GPU runtime APIs directly but
/// instead uses a small wrapper library that exports a stable and conveniently
/// typed ABI on top of GPU runtimes such as CUDA or ROCm (HIP).
///
/// A non-empty gpuBinaryAnnotation overrides the pass' command line option.
std::unique_ptr<OperationPass<ModuleOp>>
createGpuToLLVMConversionPass(StringRef gpuBinaryAnnotation = {});

/// Collect a set of patterns to convert from the GPU dialect to LLVM.
///
/// A non-empty gpuBinaryAnnotation overrides the pass' command line option.
void populateGpuToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns,
                                         StringRef gpuBinaryAnnotation = {});
} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_
