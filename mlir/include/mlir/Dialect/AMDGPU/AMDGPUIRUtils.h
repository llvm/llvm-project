//===- AMDGPUIRUtils.h - AMDGPU dialect IR utilities -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_AMDGPUIRUTILS_H
#define MLIR_DIALECT_AMDGPU_AMDGPUIRUTILS_H

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Compiler.h"

namespace mlir::amdgpu {

// Integer memory-space attributes are deprecated, but still accepted here for
// compatibility with existing IR.

inline bool isGlobalMemorySpace(Attribute memorySpace) {
  if (!memorySpace)
    return true;
  if (auto gpuMemorySpace = dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuMemorySpace.getValue() == gpu::AddressSpace::Global;
  if (LLVM_UNLIKELY(isa<IntegerAttr>(memorySpace))) {
    int64_t intMemorySpace = cast<IntegerAttr>(memorySpace).getInt();
    return intMemorySpace == 0 || intMemorySpace == 1;
  }
  return false;
}

inline bool isWorkgroupMemorySpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto gpuMemorySpace = dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuMemorySpace.getValue() == gpu::AddressSpace::Workgroup;
  if (LLVM_UNLIKELY(isa<IntegerAttr>(memorySpace))) {
    int64_t intMemorySpace = cast<IntegerAttr>(memorySpace).getInt();
    return intMemorySpace == 3;
  }
  return false;
}

inline bool isFatRawBufferMemorySpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto amdgpuMemorySpace = dyn_cast<amdgpu::AddressSpaceAttr>(memorySpace))
    return amdgpuMemorySpace.getValue() == amdgpu::AddressSpace::FatRawBuffer;
  if (LLVM_UNLIKELY(isa<IntegerAttr>(memorySpace))) {
    int64_t intMemorySpace = cast<IntegerAttr>(memorySpace).getInt();
    return intMemorySpace == 7;
  }
  return false;
}

} // namespace mlir::amdgpu

#endif // MLIR_DIALECT_AMDGPU_AMDGPUIRUTILS_H
