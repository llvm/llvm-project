//===- GPUToNVVMPass.h - Convert GPU kernel to NVVM dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_
#define MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;

template <typename OpT>
class OperationPass;

namespace gpu {
class GPUModuleOp;
class MMAMatrixType;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTONVVMOPS
#include "mlir/Conversion/Passes.h.inc"

LLVM::LLVMStructType convertMMAToLLVMType(gpu::MMAMatrixType type);

/// Configure target to convert from the GPU dialect to NVVM.
void configureGpuToNVVMConversionLegality(ConversionTarget &target);

/// Collect a set of patterns to convert from the GPU dialect to NVVM.
void populateGpuToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

/// Populate GpuSubgroupReduce pattern to NVVM. It generates a specific nvvm
/// op that is not available on every GPU.
void populateGpuSubgroupReduceOpLoweringPattern(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns);

/// Collect a set of patterns to convert WMMA ops from GPU dialect to NVVM.
void populateGpuWMMAToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);

/// Creates a pass that lowers GPU dialect operations to NVVM counterparts. The
/// index bitwidth used for the lowering of the device side index computations
/// is configurable.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createLowerGpuOpsToNVVMOpsPass(
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout,
    bool hasRedux = false);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTONVVM_GPUTONVVMPASS_H_
