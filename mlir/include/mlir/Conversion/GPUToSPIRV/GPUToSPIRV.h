//===- GPUToSPIRV.h - GPU to SPIR-V Patterns --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert GPU dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOSPIRV_GPUTOSPIRV_H
#define MLIR_CONVERSION_GPUTOSPIRV_GPUTOSPIRV_H

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating GPU Ops to
/// SPIR-V ops. For a gpu.func to be converted, it should have a
/// spirv.entry_point_abi attribute.
void populateGPUToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                RewritePatternSet &patterns);

/// Collect a set of patterns to convert WMMA ops from GPU dialect to SPIRV,
/// using the KHR Cooperative Matrix extension.
void populateGpuWMMAToSPIRVCoopMatrixKHRConversionPatterns(
    SPIRVTypeConverter &typeConverter, RewritePatternSet &patterns);

/// Adds `MMAMatrixType` conversions to SPIR-V cooperative matrix KHR type
/// conversion to the type converter.
void populateMMAToSPIRVCoopMatrixTypeConversion(
    SPIRVTypeConverter &typeConverter);
} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOSPIRV_GPUTOSPIRV_H
