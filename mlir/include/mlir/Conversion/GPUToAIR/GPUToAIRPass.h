//===- GPUToAIRPass.h - Convert GPU kernels to AIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOAIR_GPUTOAIRPASS_H_
#define MLIR_CONVERSION_GPUTOAIR_GPUTOAIRPASS_H_

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

#define GEN_PASS_DECL_CONVERTGPUOPSTOAIROPS
#include "mlir/Conversion/Passes.h.inc"

/// The data layout of the AIR target.
extern const char *const kAIRDataLayout;

void populateGpuToAIRConversionPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);

void configureGpuToAIRConversionLegality(ConversionTarget &target);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOAIR_GPUTOAIRPASS_H_
