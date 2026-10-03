//===-- llvm/TargetParser/AtomicScope.h ---Atomic Scope--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_ATOMICSCOPE_H
#define LLVM_TARGETPARSER_ATOMICSCOPE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>
#include <utility>

namespace llvm {

/// Target-neutral memory synchronization scopes.
///
/// The underlying values are ABI-sensitive and should not be changed.
enum class AtomicScope : unsigned {
  System = 0,    // __MEMORY_SCOPE_SYSTEM
  Device = 1,    // __MEMORY_SCOPE_DEVICE
  Workgroup = 2, // __MEMORY_SCOPE_WRKGRP
  Wavefront = 3, // __MEMORY_SCOPE_WVFRNT
  Single = 4,    // __MEMORY_SCOPE_SINGLE
  Cluster = 5,   // __MEMORY_SCOPE_CLUSTR
};

/// Returns the LLVM IR syncscope string that \p T uses to spell \p S.
inline std::optional<StringRef>
getAtomicScopeIRString(const Triple &T, AtomicScope S,
                       bool IsSingleAddressSpace = false) {
  if (T.isAMDGPU()) {
    switch (S) {
    case AtomicScope::Single:
      return IsSingleAddressSpace ? "singlethread-one-as" : "singlethread";
    case AtomicScope::Wavefront:
      return IsSingleAddressSpace ? "wavefront-one-as" : "wavefront";
    case AtomicScope::Workgroup:
      return IsSingleAddressSpace ? "workgroup-one-as" : "workgroup";
    case AtomicScope::Cluster:
      return IsSingleAddressSpace ? "cluster-one-as" : "cluster";
    case AtomicScope::Device:
      return IsSingleAddressSpace ? "agent-one-as" : "agent";
    case AtomicScope::System:
      return IsSingleAddressSpace ? "one-as" : "";
    }
    return std::nullopt;
  }
  if (T.isNVPTX()) {
    switch (S) {
    case AtomicScope::Single:
      return "singlethread";
    // NVPTX has no distinct wavefront/subgroup scope; it folds into block.
    case AtomicScope::Wavefront:
    case AtomicScope::Workgroup:
      return "block";
    case AtomicScope::Cluster:
      return "cluster";
    case AtomicScope::Device:
      return "device";
    case AtomicScope::System:
      return "";
    }
    return std::nullopt;
  }
  if (T.isSPIRV()) {
    switch (S) {
    case AtomicScope::Single:
      return "singlethread";
    case AtomicScope::Wavefront:
      return "subgroup";
    // SPIR-V has no cluster scope; it folds into workgroup.
    case AtomicScope::Cluster:
    case AtomicScope::Workgroup:
      return "workgroup";
    case AtomicScope::Device:
      return "device";
    case AtomicScope::System:
      return "";
    }
    return std::nullopt;
  }
  return std::nullopt;
}

/// Parses a target syncscope string into its abstract scope, the inverse of
/// getAtomicScopeIRString. Returns the scope and whether it is the single
/// address space variant.
inline std::optional<std::pair<AtomicScope, bool>>
parseAtomicScopeIRString(const Triple &T, StringRef Name) {
  using Result = std::optional<std::pair<AtomicScope, bool>>;
  auto Make = [](AtomicScope S,
                 bool IsSingleAddressSpace) -> std::pair<AtomicScope, bool> {
    return {S, IsSingleAddressSpace};
  };
  if (T.isAMDGPU())
    return StringSwitch<Result>(Name)
        .Case("singlethread", Make(AtomicScope::Single, false))
        .Case("wavefront", Make(AtomicScope::Wavefront, false))
        .Case("workgroup", Make(AtomicScope::Workgroup, false))
        .Case("cluster", Make(AtomicScope::Cluster, false))
        .Case("agent", Make(AtomicScope::Device, false))
        .Case("", Make(AtomicScope::System, false))
        .Case("singlethread-one-as", Make(AtomicScope::Single, true))
        .Case("wavefront-one-as", Make(AtomicScope::Wavefront, true))
        .Case("workgroup-one-as", Make(AtomicScope::Workgroup, true))
        .Case("cluster-one-as", Make(AtomicScope::Cluster, true))
        .Case("agent-one-as", Make(AtomicScope::Device, true))
        .Case("one-as", Make(AtomicScope::System, true))
        .Default(std::nullopt);
  if (T.isNVPTX())
    return StringSwitch<Result>(Name)
        .Case("singlethread", Make(AtomicScope::Single, false))
        .Case("block", Make(AtomicScope::Workgroup, false))
        .Case("cluster", Make(AtomicScope::Cluster, false))
        .Case("device", Make(AtomicScope::Device, false))
        .Case("", Make(AtomicScope::System, false))
        .Default(std::nullopt);
  if (T.isSPIRV())
    return StringSwitch<Result>(Name)
        .Case("singlethread", Make(AtomicScope::Single, false))
        .Case("subgroup", Make(AtomicScope::Wavefront, false))
        .Case("workgroup", Make(AtomicScope::Workgroup, false))
        .Case("device", Make(AtomicScope::Device, false))
        .Case("", Make(AtomicScope::System, false))
        .Default(std::nullopt);
  return std::nullopt;
}

} // end namespace llvm

#endif // LLVM_TARGETPARSER_ATOMICSCOPE_H
