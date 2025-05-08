//===--- AMDGPUMachineModuleInfo.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU Machine Module Info.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEMODULEINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEMODULEINFO_H

#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {

class AMDGPUMachineModuleInfo final : public MachineModuleInfoELF {
private:

  // All supported memory/synchronization scopes can be found here:
  //   http://llvm.org/docs/AMDGPUUsage.html#memory-scopes

  /// Agent synchronization scope ID (cross address space).
  SyncScope::ID AgentSSID;
  /// Workgroup synchronization scope ID (cross address space).
  SyncScope::ID WorkgroupSSID;
  /// Wavefront synchronization scope ID (cross address space).
  SyncScope::ID WavefrontSSID;
#if LLPC_BUILD_NPI
  /// Cluster synchronization scope ID (cross address space).
  SyncScope::ID ClusterSSID;
#endif /* LLPC_BUILD_NPI */
  /// System synchronization scope ID (single address space).
  SyncScope::ID SystemOneAddressSpaceSSID;
  /// Agent synchronization scope ID (single address space).
  SyncScope::ID AgentOneAddressSpaceSSID;
  /// Workgroup synchronization scope ID (single address space).
  SyncScope::ID WorkgroupOneAddressSpaceSSID;
  /// Wavefront synchronization scope ID (single address space).
  SyncScope::ID WavefrontOneAddressSpaceSSID;
  /// Single thread synchronization scope ID (single address space).
  SyncScope::ID SingleThreadOneAddressSpaceSSID;
#if LLPC_BUILD_NPI
  /// Cluster synchronization scope ID (single address space).
  SyncScope::ID ClusterOneAddressSpaceSSID;
#endif /* LLPC_BUILD_NPI */

  /// In AMDGPU target synchronization scopes are inclusive, meaning a
  /// larger synchronization scope is inclusive of a smaller synchronization
  /// scope.
  ///
  /// \returns \p SSID's inclusion ordering, or "std::nullopt" if \p SSID is not
  /// supported by the AMDGPU target.
  std::optional<uint8_t>
  getSyncScopeInclusionOrdering(SyncScope::ID SSID) const {
    if (SSID == SyncScope::SingleThread ||
        SSID == getSingleThreadOneAddressSpaceSSID())
      return 0;
    else if (SSID == getWavefrontSSID() ||
             SSID == getWavefrontOneAddressSpaceSSID())
      return 1;
    else if (SSID == getWorkgroupSSID() ||
             SSID == getWorkgroupOneAddressSpaceSSID())
      return 2;
#if LLPC_BUILD_NPI
    else if (SSID == getClusterSSID() ||
             SSID == getClusterOneAddressSpaceSSID())
      return 3;
#endif /* LLPC_BUILD_NPI */
    else if (SSID == getAgentSSID() ||
             SSID == getAgentOneAddressSpaceSSID())
#if LLPC_BUILD_NPI
      return 4;
#else /* LLPC_BUILD_NPI */
      return 3;
#endif /* LLPC_BUILD_NPI */
    else if (SSID == SyncScope::System ||
             SSID == getSystemOneAddressSpaceSSID())
#if LLPC_BUILD_NPI
      return 5;
#else /* LLPC_BUILD_NPI */
      return 4;
#endif /* LLPC_BUILD_NPI */

    return std::nullopt;
  }

  /// \returns True if \p SSID is restricted to single address space, false
  /// otherwise
  bool isOneAddressSpace(SyncScope::ID SSID) const {
#if LLPC_BUILD_NPI
    return SSID == getClusterOneAddressSpaceSSID() ||
           SSID == getSingleThreadOneAddressSpaceSSID() ||
           SSID == getWavefrontOneAddressSpaceSSID() ||
           SSID == getWorkgroupOneAddressSpaceSSID() ||
           SSID == getAgentOneAddressSpaceSSID() ||
           SSID == getSystemOneAddressSpaceSSID();
#else /* LLPC_BUILD_NPI */
    return SSID == getSingleThreadOneAddressSpaceSSID() ||
        SSID == getWavefrontOneAddressSpaceSSID() ||
        SSID == getWorkgroupOneAddressSpaceSSID() ||
        SSID == getAgentOneAddressSpaceSSID() ||
        SSID == getSystemOneAddressSpaceSSID();
#endif /* LLPC_BUILD_NPI */
  }

public:
  AMDGPUMachineModuleInfo(const MachineModuleInfo &MMI);

  /// \returns Agent synchronization scope ID (cross address space).
  SyncScope::ID getAgentSSID() const {
    return AgentSSID;
  }
  /// \returns Workgroup synchronization scope ID (cross address space).
  SyncScope::ID getWorkgroupSSID() const {
    return WorkgroupSSID;
  }
  /// \returns Wavefront synchronization scope ID (cross address space).
  SyncScope::ID getWavefrontSSID() const {
    return WavefrontSSID;
  }
#if LLPC_BUILD_NPI
  /// \returns Cluster synchronization scope ID (cross address space).
  SyncScope::ID getClusterSSID() const { return ClusterSSID; }
#endif /* LLPC_BUILD_NPI */
  /// \returns System synchronization scope ID (single address space).
  SyncScope::ID getSystemOneAddressSpaceSSID() const {
    return SystemOneAddressSpaceSSID;
  }
  /// \returns Agent synchronization scope ID (single address space).
  SyncScope::ID getAgentOneAddressSpaceSSID() const {
    return AgentOneAddressSpaceSSID;
  }
  /// \returns Workgroup synchronization scope ID (single address space).
  SyncScope::ID getWorkgroupOneAddressSpaceSSID() const {
    return WorkgroupOneAddressSpaceSSID;
  }
  /// \returns Wavefront synchronization scope ID (single address space).
  SyncScope::ID getWavefrontOneAddressSpaceSSID() const {
    return WavefrontOneAddressSpaceSSID;
  }
  /// \returns Single thread synchronization scope ID (single address space).
  SyncScope::ID getSingleThreadOneAddressSpaceSSID() const {
    return SingleThreadOneAddressSpaceSSID;
#if LLPC_BUILD_NPI
  }
  /// \returns Single thread synchronization scope ID (single address space).
  SyncScope::ID getClusterOneAddressSpaceSSID() const {
    return ClusterOneAddressSpaceSSID;
#endif /* LLPC_BUILD_NPI */
  }

  /// In AMDGPU target synchronization scopes are inclusive, meaning a
  /// larger synchronization scope is inclusive of a smaller synchronization
  /// scope.
  ///
  /// \returns True if synchronization scope \p A is larger than or equal to
  /// synchronization scope \p B, false if synchronization scope \p A is smaller
  /// than synchronization scope \p B, or "std::nullopt" if either
  /// synchronization scope \p A or \p B is not supported by the AMDGPU target.
  std::optional<bool> isSyncScopeInclusion(SyncScope::ID A,
                                           SyncScope::ID B) const {
    const auto &AIO = getSyncScopeInclusionOrdering(A);
    const auto &BIO = getSyncScopeInclusionOrdering(B);
    if (!AIO || !BIO)
      return std::nullopt;

    bool IsAOneAddressSpace = isOneAddressSpace(A);
    bool IsBOneAddressSpace = isOneAddressSpace(B);

    return *AIO >= *BIO &&
           (IsAOneAddressSpace == IsBOneAddressSpace || !IsAOneAddressSpace);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEMODULEINFO_H
