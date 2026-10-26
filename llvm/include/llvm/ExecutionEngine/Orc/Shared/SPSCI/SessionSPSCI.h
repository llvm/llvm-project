//===------ SessionSPSCI.h - SPS CI for runtime Session APIs ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS controller-interface descriptors for the runtime's Session interface.
// Unlike most controller-interface operations, these take no instance address:
// they act on the runtime Session that receives the call.
//
// See CallSPSCI.h for a description of the descriptor scheme.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SESSIONSPSCI_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SESSIONSPSCI_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolLookupSetSPS.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"

namespace llvm::orc::rt::sps_ci {

/// Resolve the given lookup set against the Session's instance symbols: names
/// for objects created at runtime, which have no symbols under the standard
/// object models. Returns one result per element of the set, in order.
struct SessionLookupInstanceSymbols {
  static constexpr SymbolNameSpec Name =
      SymbolNameSpec::c("orc_rt_ci_sps_Session_lookupInstanceSymbols");
  using SPSSig =
      shared::SPSSequence<shared::SPSOptional<shared::SPSExecutorAddr>>(
          shared::SPSSymbolLookupSet);
};

} // namespace llvm::orc::rt::sps_ci

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SPSCI_SESSIONSPSCI_H
