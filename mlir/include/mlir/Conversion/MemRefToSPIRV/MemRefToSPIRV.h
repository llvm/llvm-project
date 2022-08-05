//===- MemRefToSPIRV.h - MemRef to SPIR-V Patterns --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert MemRef dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRV_H
#define MLIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRV_H

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
class SPIRVTypeConverter;

namespace spirv {
/// Mapping from numeric MemRef memory spaces into SPIR-V symbolic ones.
using MemorySpaceToStorageClassMap = DenseMap<unsigned, spirv::StorageClass>;
/// Returns the default map for targeting Vulkan-flavored SPIR-V.
MemorySpaceToStorageClassMap getDefaultVulkanStorageClassMap();

/// Type converter for converting numeric MemRef memory spaces into SPIR-V
/// symbolic ones.
class MemorySpaceToStorageClassConverter : public TypeConverter {
public:
  explicit MemorySpaceToStorageClassConverter(
      const MemorySpaceToStorageClassMap &memorySpaceMap);

private:
  const MemorySpaceToStorageClassMap &memorySpaceMap;
};

/// Creates the target that populates legality of ops with MemRef types.
std::unique_ptr<ConversionTarget>
getMemorySpaceToStorageClassTarget(MLIRContext &);

/// Appends to a pattern list additional patterns for converting numeric MemRef
/// memory spaces into SPIR-V symbolic ones.
void populateMemorySpaceToStorageClassPatterns(
    MemorySpaceToStorageClassConverter &typeConverter,
    RewritePatternSet &patterns);

} // namespace spirv

/// Appends to a pattern list additional patterns for translating MemRef ops
/// to SPIR-V ops.
void populateMemRefToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                   RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOSPIRV_MEMREFTOSPIRV_H
