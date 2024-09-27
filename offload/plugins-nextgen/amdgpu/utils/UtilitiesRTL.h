//===----RTLs/amdgpu/utils/UtilitiesRTL.h ------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL Utilities for AMDGPU plugins
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "Shared/Debug.h"
#include "Utils/ELF.h"

#include "omptarget.h"

#include "llvm/Frontend/Offloading/Utility.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {
namespace hsa_utils {

// The implicit arguments of COV5 AMDGPU kernels.
struct AMDGPUImplicitArgsTy {
  uint32_t BlockCountX;
  uint32_t BlockCountY;
  uint32_t BlockCountZ;
  uint16_t GroupSizeX;
  uint16_t GroupSizeY;
  uint16_t GroupSizeZ;
  uint8_t Unused0[46]; // 46 byte offset.
  uint16_t GridDims;
  uint8_t Unused1[54]; // 54 byte offset.
  uint32_t DynamicLdsSize;
  uint8_t Unused2[132]; // 132 byte offset.
};

// Dummy struct for COV4 implicitargs.
struct AMDGPUImplicitArgsTyCOV4 {
  uint8_t Unused[56];
};

/// Returns the size in bytes of the implicit arguments of AMDGPU kernels.
/// `Version` is the ELF ABI version, e.g. COV5.
inline uint32_t getImplicitArgsSize(uint16_t Version) {
  return Version < ELF::ELFABIVERSION_AMDGPU_HSA_V5
             ? sizeof(AMDGPUImplicitArgsTyCOV4)
             : sizeof(AMDGPUImplicitArgsTy);
}

/// Reads the AMDGPU specific metadata from the ELF file and propagates the
/// KernelInfoMap
inline Error readAMDGPUMetaDataFromImage(
    MemoryBufferRef MemBuffer,
    StringMap<offloading::amdgpu::AMDGPUKernelMetaData> &KernelInfoMap,
    uint16_t &ELFABIVersion) {
  Error Err = llvm::offloading::amdgpu::getAMDGPUMetaDataFromImage(
      MemBuffer, KernelInfoMap, ELFABIVersion);
  if (!Err)
    return Err;
  DP("ELFABIVERSION Version: %u\n", ELFABIVersion);
  return Err;
}

} // namespace hsa_utils
} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
