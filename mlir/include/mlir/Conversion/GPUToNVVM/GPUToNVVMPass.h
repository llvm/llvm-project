//===- GPUToNVVMPass.h - Convert GPU kernel to NVVM dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_
#define MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class OwningRewritePatternList;
class ConversionTarget;

template <typename OpT> class OperationPass;

namespace gpu {
class GPUModuleOp;
}

/// Configure target to convert from the GPU dialect to NVVM.
void configureGpuToNVVMConversionLegality(ConversionTarget &target);

/// Collect a set of patterns to convert from the GPU dialect to NVVM.
void populateGpuToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns);

/// Creates a pass that lowers GPU dialect operations to NVVM counterparts. The
/// index bitwidth used for the lowering of the device side index computations
/// is configurable.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createLowerGpuOpsToNVVMOpsPass(
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_
