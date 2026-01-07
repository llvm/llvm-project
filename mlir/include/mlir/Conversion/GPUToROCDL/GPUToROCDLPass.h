//===- GPUToROCDLPass.h - Convert GPU kernel to ROCDL dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
#define MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_

#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include <memory>

namespace mlir {
class Pass;
class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;

template <typename OpT>
class OperationPass;

namespace amdgpu {
struct Chipset;
} // namespace amdgpu

namespace gpu {
class GPUModuleOp;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTOROCDLOPS
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the GPU dialect to ROCDL.
/// If `runtime` is Unknown, gpu.printf will not be lowered
/// The resulting pattern set should be run over a gpu.module op
void populateGpuToROCDLConversionPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          gpu::amd::Runtime runtime,
                                          amdgpu::Chipset chipset);

/// Configure target to convert from the GPU dialect to ROCDL.
void configureGpuToROCDLConversionLegality(ConversionTarget &target);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
