//===- GPUToGENXPass.h - Convert GPU dialect to GENX dialect ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOGENX_GPUTOGENXPASS_H_
#define MLIR_CONVERSION_GPUTOGENX_GPUTOGENXPASS_H_

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;

template <typename OpT>
class OperationPass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTOGENXOPS
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the GPU dialect to the GENX
/// dialect.
void populateGpuToGENXConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

/// Configure target to convert from the GPU dialect to the GENX dialect.
void configureGpuToGENXConversionLegality(ConversionTarget &target);

/// Creates a pass that lowers GPU dialect operations to GENX counterparts. The
/// index bitwidth used for the lowering of the device side index computations
/// is configurable.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createLowerGpuOpsToGENXOpsPass(
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOGENX_GPUTOGENXPASS_H_
