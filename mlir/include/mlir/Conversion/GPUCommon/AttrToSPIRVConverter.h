//===- AttrToSPIRVConverter.h - GPU attributes conversion to SPIR-V - C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_ATTRTOSPIRVCONVERTER_H_
#define MLIR_CONVERSION_GPUCOMMON_ATTRTOSPIRVCONVERTER_H_

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVEnums.h>

namespace mlir {
spirv::StorageClass addressSpaceToStorageClass(gpu::AddressSpace addressSpace);
} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_ATTRTOSPIRVCONVERTER_H_
