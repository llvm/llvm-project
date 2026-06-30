//===- PtrToSPIRV.h - Convert Ptr to SPIR-V dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PTRTOSPIRV_PTRTOSPIRV_H
#define MLIR_CONVERSION_PTRTOSPIRV_PTRTOSPIRV_H

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class RewritePatternSet;
class SPIRVTypeConverter;

#define GEN_PASS_DECL_CONVERTPTRTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"

namespace ptr {
/// Populates the type converter with conversions for ptr dialect types.
void populatePtrToSPIRVTypeConversions(SPIRVTypeConverter &typeConverter);

/// Appends patterns for lowering ptr dialect operations to SPIR-V operations.
void populatePtrToSPIRVPatterns(const SPIRVTypeConverter &typeConverter,
                                RewritePatternSet &patterns,
                                spirv::StorageClass storageClass =
                                    spirv::StorageClass::PhysicalStorageBuffer);

std::unique_ptr<OperationPass<>> createConvertPtrToSPIRVPass();
} // namespace ptr
} // namespace mlir

#endif // MLIR_CONVERSION_PTRTOSPIRV_PTRTOSPIRV_H
