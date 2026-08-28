//===--- NativeDylibManagerSPSCI.h - SPS CI for dylib mgmt ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS controller-interface descriptors for the runtime's NativeDylibManager
// interface. The instance address of the executor-side manager is passed as
// the first argument to each call.
//
// See CallSPSCI.h for a description of the descriptor scheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_NATIVEDYLIBMANAGERSPSCI_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_NATIVEDYLIBMANAGERSPSCI_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"

#include <cstdint>

namespace llvm::orc::rt::sps_ci {

/// The executor-side dylib-manager instance. This is a data symbol -- passed
/// as the first argument to each call below -- not a wrapper to call.
inline constexpr char NativeDylibManagerInstanceName[] =
    "orc_rt_ci_NativeDylibManager_Instance";

/// Open the dylib at the given path with the given mode flags; returns a
/// handle to it.
struct DylibMgrOpen {
  static constexpr char Name[] = "orc_rt_ci_sps_NativeDylibManager_load";
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
      shared::SPSExecutorAddr, shared::SPSString, uint64_t);
};

/// Resolve the given lookup set within the given dylib.
struct DylibMgrResolve {
  static constexpr char Name[] = "orc_rt_ci_sps_NativeDylibManager_lookup";
  using SPSSig = shared::SPSExpected<
      shared::SPSSequence<shared::SPSOptional<shared::SPSExecutorAddr>>>(
      shared::SPSExecutorAddr, shared::SPSExecutorAddr,
      shared::SPSRemoteSymbolLookupSet);
};

} // namespace llvm::orc::rt::sps_ci

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_NATIVEDYLIBMANAGERSPSCI_H
