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

/// A list of offsets required by the ABI of code object versions 4 and 5.
enum COV_OFFSETS : uint32_t {
  // 128 KB
  PER_DEVICE_PREALLOC_SIZE = 131072
};

typedef unsigned XnackBuildMode;

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
  uint8_t Unused1[30]; // 30 byte offset.
  uint64_t HeapV1Ptr;
  uint8_t Unused2[16]; // 16 byte offset.
  uint32_t DynamicLdsSize;
  uint8_t Unused3[132]; // 132 byte offset.
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

// Check target image for XNACK mode (XNACK+, XNACK-ANY, XNACK-)
[[nodiscard]] XnackBuildMode
extractXnackModeFromBinary(const __tgt_device_image *TgtImage) {
  assert((TgtImage != nullptr) && "TgtImage is nullptr.");
  StringRef Buffer(reinterpret_cast<const char *>(TgtImage->ImageStart),
                   utils::getPtrDiff(TgtImage->ImageEnd, TgtImage->ImageStart));
  auto ElfOrErr =
      ELF64LEObjectFile::create(MemoryBufferRef(Buffer, /*Identifier=*/""),
                                /*InitContent=*/false);
  if (auto Err = ElfOrErr.takeError()) {
    consumeError(std::move(Err));
    DP("An error occured while reading ELF to extract XNACK mode\n");
    return ELF::EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4;
  }
  u_int16_t EFlags = ElfOrErr->getPlatformFlags();

  hsa_utils::XnackBuildMode XnackFlags = EFlags & ELF::EF_AMDGPU_FEATURE_XNACK_V4;

  if (XnackFlags == ELF::EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4)
    DP("XNACK is not supported on this system!\n");

  return XnackFlags;
}

void checkImageCompatibilityWithSystemXnackMode(__tgt_device_image *TgtImage,
                                                bool IsXnackEnabled) {
  hsa_utils::XnackBuildMode ImageXnackMode =
      hsa_utils::extractXnackModeFromBinary(TgtImage);

  if (ImageXnackMode == ELF::EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4)
    return;

  if (IsXnackEnabled &&
      (ImageXnackMode == ELF::EF_AMDGPU_FEATURE_XNACK_OFF_V4)) {
    FAILURE_MESSAGE(
        "Image is not compatible with current XNACK mode! XNACK is enabled "
        "on the system but image was compiled with xnack-.\n");
  } else if (!IsXnackEnabled &&
             (ImageXnackMode == ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4)) {
    FAILURE_MESSAGE("Image is not compatible with current XNACK mode! "
                    "XNACK is disabled on the system. However, the image "
                    "requires xnack+.\n");
  }
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
