//===---- ObjectUtilities.h - AMDGPU ELF utilities ---------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares AMDGPU ELF related utilities.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace llvm {
namespace offloading {
namespace amdgpu {
/// Check if an image is compatible with current system's environment. The
/// system environment is given as a 'target-id' which has the form:
///
/// <target-id> := <processor> ( ":" <target-feature> ( "+" | "-" ) )*
///
/// If a feature is not specific as '+' or '-' it is assumed to be in an 'any'
/// and is compatible with either '+' or '-'. The HSA runtime returns this
/// information using the target-id, while we use the ELF header to determine
/// these features.
bool isImageCompatibleWithEnv(StringRef ImageArch, uint32_t ImageFlags,
                              StringRef EnvTargetID);

/// Struct for holding metadata related to AMDGPU kernels, for more information
/// about the metadata and its meaning see:
/// https://llvm.org/docs/AMDGPUUsage.html#code-object-v3
struct AMDGPUKernelMetaData {
  /// Constant indicating that a value is invalid.
  static constexpr uint32_t KInvalidValue =
      std::numeric_limits<uint32_t>::max();
  /// The amount of group segment memory required by a work-group in bytes.
  uint32_t GroupSegmentList = KInvalidValue;
  /// The amount of fixed private address space memory required for a work-item
  /// in bytes.
  uint32_t PrivateSegmentSize = KInvalidValue;
  /// Number of scalar registers required by a wavefront.
  uint32_t SGPRCount = KInvalidValue;
  /// Number of vector registers required by each work-item.
  uint32_t VGPRCount = KInvalidValue;
  /// Number of stores from a scalar register to a register allocator created
  /// spill location.
  uint32_t SGPRSpillCount = KInvalidValue;
  /// Number of stores from a vector register to a register allocator created
  /// spill location.
  uint32_t VGPRSpillCount = KInvalidValue;
  /// Number of accumulator registers required by each work-item.
  uint32_t AGPRCount = KInvalidValue;
  /// Corresponds to the OpenCL reqd_work_group_size attribute.
  uint32_t RequestedWorkgroupSize[3] = {KInvalidValue, KInvalidValue,
                                        KInvalidValue};
  /// Corresponds to the OpenCL work_group_size_hint attribute.
  uint32_t WorkgroupSizeHint[3] = {KInvalidValue, KInvalidValue, KInvalidValue};
  /// Wavefront size.
  uint32_t WavefrontSize = KInvalidValue;
  /// Maximum flat work-group size supported by the kernel in work-items.
  uint32_t MaxFlatWorkgroupSize = KInvalidValue;
};

/// Reads AMDGPU specific metadata from the ELF file and propagates the
/// KernelInfoMap.
Error getAMDGPUMetaDataFromImage(MemoryBufferRef MemBuffer,
                                 StringMap<AMDGPUKernelMetaData> &KernelInfoMap,
                                 uint16_t &ELFABIVersion);
} // namespace amdgpu
} // namespace offloading
} // namespace llvm
