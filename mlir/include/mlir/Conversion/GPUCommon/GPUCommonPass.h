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

/// Creates a pass to convert kernel functions into GPU target object blobs.
///
/// This transformation takes the body of each function that is annotated with
/// the 'gpu.kernel' attribute, copies it to a new LLVM module, compiles the
/// module with help of the GPU backend to target object and then invokes
/// the provided blobGenerator to produce a binary blob. Such blob is then
/// attached as a string attribute to the kernel function.
///
/// Following callbacks are to be provided by user:
/// - loweringCallback : lower the module to an LLVM module.
/// - blobGenerator : build a blob executable on target GPU.
///
/// Information wrt LLVM backend are to be supplied by user:
/// - triple : target triple to be used.
/// - targetChip : mcpu to be used.
/// - features : target-specific features to be used.
///
/// Information about result attribute is to be specified by user:
/// - gpuBinaryAnnotation : the name of the attribute which contains the blob.
///
/// After the transformation, the body of the kernel function is removed (i.e.,
/// it is turned into a declaration).
///
/// A non-empty gpuBinaryAnnotation overrides the pass' command line option.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createConvertGPUKernelToBlobPass(LoweringCallback loweringCallback,
                                 BlobGenerator blobGenerator, StringRef triple,
                                 StringRef targetChip, StringRef features,
                                 StringRef gpuBinaryAnnotation = {});

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_GPUCOMMONPASS_H_
