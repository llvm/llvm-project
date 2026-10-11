//===- NativeDylibManagerSPSCI.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPS Controller Interface implementation for NativeDylibManager.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/NativeDylibManagerSPSCI.h"
#include "orc-rt/bedrock/NativeDylibManager.h"
#include "orc-rt/support/sps/SPSSymbolLookupSet.h"

namespace orc_rt::sps_ci {

ORC_RT_SPS_WRAPPER_IMPL(
    orc_rt_ci_sps_NativeDylibManager_load,
    SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::load))

ORC_RT_SPS_WRAPPER_IMPL(
    orc_rt_ci_sps_NativeDylibManager_lookup,
    SPSExpected<SPSSequence<SPSOptional<SPSExecutorAddr>>>(SPSExecutorAddr,
                                                           SPSExecutorAddr,
                                                           SPSSymbolLookupSet),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::lookup))

static std::pair<SymbolNameSpec, const void *>
    orc_rt_ci_NativeDylibManager_sps_interface[] = {
        ORC_RT_SYMTAB_C_PAIR(orc_rt_ci_sps_NativeDylibManager_load),
        ORC_RT_SYMTAB_C_PAIR(orc_rt_ci_sps_NativeDylibManager_lookup)};

Error addNativeDylibManager(SimpleSymbolTable &ST) {
  return ST.addUnique(orc_rt_ci_NativeDylibManager_sps_interface);
}

} // namespace orc_rt::sps_ci
