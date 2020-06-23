//===- GPUToNVVMPass.h - Convert GPU kernel to NVVM dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_
#define MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class OwningRewritePatternList;

template <typename OpT> class OperationPass;

namespace gpu {
class GPUModuleOp;
}

/// Collect a set of patterns to convert from the GPU dialect to NVVM.
void populateGpuToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns);

/// Creates a pass that lowers GPU dialect operations to NVVM counterparts.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createLowerGpuOpsToNVVMOpsPass();

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_
