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
    // Ordered from smallest to largest scope. Level is the index.
    // Cross-AS and one-AS scopes share the same inclusion ordering level.
    // Level | Cross-AS scope   | One-AS scope
    // ------+------------------+----------------------
    //   0   | singlethread     | singlethread-one-as
    //   1   | wavefront        | wavefront-one-as
    //   2   | workgroup        | workgroup-one-as
    //   3   | cluster          | cluster-one-as
    //   4   | agent            | agent-one-as
    //   5   | system           | one-as
    const SyncScope::ID CrossAS[] = {
        SyncScope::SingleThread, getWavefrontSSID(), getWorkgroupSSID(),
        getClusterSSID(),        getAgentSSID(),     SyncScope::System};
    const SyncScope::ID OneAS[] = {
        getSingleThreadOneAddressSpaceSSID(), getWavefrontOneAddressSpaceSSID(),
        getWorkgroupOneAddressSpaceSSID(),    getClusterOneAddressSpaceSSID(),
        getAgentOneAddressSpaceSSID(),        getSystemOneAddressSpaceSSID()};

    // Returns {level, isOneAS} for a given scope, or nullopt if unsupported.
    auto GetLevelAndOneAS =
        [&](SyncScope::ID SSID) -> std::optional<std::pair<unsigned, bool>> {
      for (auto [I, Cross, One] : llvm::enumerate(CrossAS, OneAS)) {
        if (Cross == SSID)
          return std::make_pair(I, false);
        if (One == SSID)
          return std::make_pair(I, true);
      }
      return std::nullopt;
    };

    auto AI = GetLevelAndOneAS(A);
    auto BI = GetLevelAndOneAS(B);
    if (!AI || !BI)
      return std::nullopt;

    unsigned Level = std::max(AI->first, BI->first);
    // If either scope is cross-AS, the result must be cross-AS.
    return (AI->second && BI->second) ? OneAS[Level] : CrossAS[Level];
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEMODULEINFO_H
