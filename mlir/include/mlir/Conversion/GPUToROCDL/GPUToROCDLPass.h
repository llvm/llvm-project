//===- GPUToROCDLPass.h - Convert GPU kernel to ROCDL dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
#define MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class OwningRewritePatternList;
class ConversionTarget;

template <typename OpT>
class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

/// Collect a set of patterns to convert from the GPU dialect to ROCDL.
void populateGpuToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                          OwningRewritePatternList &patterns);

/// Configure target to convert from the GPU dialect to ROCDL.
void configureGpuToROCDLConversionLegality(ConversionTarget &target);

/// Creates a pass that lowers GPU dialect operations to ROCDL counterparts. The
/// index bitwidth used for the lowering of the device side index computations
/// is configurable.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createLowerGpuOpsToROCDLOpsPass(
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
