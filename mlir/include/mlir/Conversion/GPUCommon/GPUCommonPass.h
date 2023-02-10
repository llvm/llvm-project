//===- GPUCommonPass.h - MLIR GPU runtime support -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_
#define MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_

#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
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
class RewritePatternSet;

class Pass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

#define GEN_PASS_DECL_GPUTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

using OwnedBlob = std::unique_ptr<std::vector<char>>;
using BlobGenerator =
    std::function<OwnedBlob(const std::string &, Location, StringRef)>;
using LoweringCallback = std::function<std::unique_ptr<llvm::Module>(
    Operation *, llvm::LLVMContext &, StringRef)>;

/// Collect a set of patterns to convert from the GPU dialect to LLVM and
/// populate converter for gpu types.
void populateGpuToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns,
                                         StringRef gpuBinaryAnnotation = {},
                                         bool kernelBarePtrCallConv = false);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_
