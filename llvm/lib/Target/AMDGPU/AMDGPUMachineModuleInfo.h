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
  /// Cluster synchronization scope ID (cross address space).
  SyncScope::ID ClusterSSID;
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
  /// Cluster synchronization scope ID (single address space).
  SyncScope::ID ClusterOneAddressSpaceSSID;

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
    else if (SSID == getClusterSSID() ||
             SSID == getClusterOneAddressSpaceSSID())
      return 3;
    else if (SSID == getAgentSSID() ||
             SSID == getAgentOneAddressSpaceSSID())
      return 4;
    else if (SSID == SyncScope::System ||
             SSID == getSystemOneAddressSpaceSSID())
      return 5;

    return std::nullopt;
  }

  /// \returns True if \p SSID is restricted to single address space, false
  /// otherwise
  bool isOneAddressSpace(SyncScope::ID SSID) const {
    return SSID == getClusterOneAddressSpaceSSID() ||
           SSID == getSingleThreadOneAddressSpaceSSID() ||
           SSID == getWavefrontOneAddressSpaceSSID() ||
           SSID == getWorkgroupOneAddressSpaceSSID() ||
           SSID == getAgentOneAddressSpaceSSID() ||
           SSID == getSystemOneAddressSpaceSSID();
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
  /// \returns Cluster synchronization scope ID (cross address space).
  SyncScope::ID getClusterSSID() const { return ClusterSSID; }
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
  }
  /// \returns Single thread synchronization scope ID (single address space).
  SyncScope::ID getClusterOneAddressSpaceSSID() const {
    return ClusterOneAddressSpaceSSID;
  }

  /// In AMDGPU, synchronization scopes are inclusive: a larger scope is
  /// inclusive of a smaller one (e.g. agent includes workgroup).
  ///
  /// Returns the merged synchronization scope of \p A and \p B: the smallest
  /// scope that is inclusive of both. Takes the larger inclusion level and,
  /// if either scope is cross-address-space, the result is also
  /// cross-address-space (since a one-AS scope cannot subsume a cross-AS
  /// scope at the same level).
  ///
  /// \returns The merged scope ID, or "std::nullopt" if either scope is not
  /// supported by the AMDGPU target.
  std::optional<SyncScope::ID> getMergedSyncScopeID(SyncScope::ID A,
                                                    SyncScope::ID B) const {
    const auto &AIO = getSyncScopeInclusionOrdering(A);
    const auto &BIO = getSyncScopeInclusionOrdering(B);
    if (!AIO || !BIO)
      return std::nullopt;

    uint8_t Level = std::max(*AIO, *BIO);
    // If either scope is cross-AS, the result must be cross-AS.
    if (isOneAddressSpace(A) && isOneAddressSpace(B)) {
      switch (Level) {
      case 0:
        return SyncScope::SingleThread;
      case 1:
        return getWavefrontOneAddressSpaceSSID();
      case 2:
        return getWorkgroupOneAddressSpaceSSID();
      case 3:
        return getClusterOneAddressSpaceSSID();
      case 4:
        return getAgentOneAddressSpaceSSID();
      case 5:
        return getSystemOneAddressSpaceSSID();
      }
    } else {
      switch (Level) {
      case 0:
        return SyncScope::SingleThread;
      case 1:
        return getWavefrontSSID();
      case 2:
        return getWorkgroupSSID();
      case 3:
        return getClusterSSID();
      case 4:
        return getAgentSSID();
      case 5:
        return SyncScope::System;
      }
    }
    return std::nullopt;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEMODULEINFO_H
