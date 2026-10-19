//===- GPUToAGXIRPass.h - Convert GPU kernel to AGXIR dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOAGXIR_GPUTOAGXIRPASS_H_
#define MLIR_CONVERSION_GPUTOAGXIR_GPUTOAGXIRPASS_H_

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;
class Pass;

namespace gpu {
class GPUModuleOp;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTOAGXIROPS
#include "mlir/Conversion/Passes.h.inc"

/// The data layout of the AGX backend.
extern const char *const kAGXDataLayout;

/// Collect a set of patterns to convert from the GPU dialect to AGXIR.
void populateGpuToAGXIRConversionPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

/// Configure target to convert from the GPU dialect to AGXIR.
void configureGpuToAGXIRConversionLegality(ConversionTarget &target);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOAGXIR_GPUTOAGXIRPASS_H_
