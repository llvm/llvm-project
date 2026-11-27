//===--- NativeDylibManagerSPSCI.h - SPS CI for dylib mgmt ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS controller-interface descriptors for the runtime's NativeDylibManager
// interface. Instances of the executor-side manager are created by calling
// DylibMgrCreate, and the instance address is passed as the first argument to
// each of the other calls.
//
// See CallSPSCI.h for a description of the descriptor scheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_NATIVEDYLIBMANAGERSPSCI_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_NATIVEDYLIBMANAGERSPSCI_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolLookupSetSPS.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"

#include <cstdint>

namespace llvm::orc::rt::sps_ci {

/// Create a new executor-side dylib-manager instance; returns its address.
/// The instance is owned by the executor, and lives until the executor shuts
/// down.
struct DylibMgrCreate {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_ci_sps_NativeDylibManager_create");
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>();
};

/// A pre-created executor-side dylib-manager instance. This is a data symbol
/// (the manager object), not a wrapper to call.
///
/// This is published by executors that do not provide DylibMgrCreate --
/// currently only LLVM's OrcTargetProcess library, whose wrapper functions
/// have no context through which a new instance could be registered. Executors
/// built on the ORC runtime provide DylibMgrCreate instead.
///
/// FIXME: Remove this once OrcTargetProcess is retired.
inline constexpr SymbolNameSpec NativeDylibManagerInstanceName =
    SymbolNameSpec::c("orc_rt_ci_NativeDylibManager_Instance");

/// Open the dylib at the given path with the given mode flags; returns a
/// handle to it.
struct DylibMgrOpen {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_ci_sps_NativeDylibManager_load");
  using SPSSig = shared::SPSExpected<shared::SPSExecutorAddr>(
      shared::SPSExecutorAddr, shared::SPSString, uint64_t);
};

/// Resolve the given lookup set within the given dylib.
struct DylibMgrResolve {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_ci_sps_NativeDylibManager_lookup");
  using SPSSig = shared::SPSExpected<
      shared::SPSSequence<shared::SPSOptional<shared::SPSExecutorAddr>>>(
      shared::SPSExecutorAddr, shared::SPSExecutorAddr,
      shared::SPSSymbolLookupSet);
};

} // namespace llvm::orc::rt::sps_ci

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_NATIVEDYLIBMANAGERSPSCI_H
