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

#include "orc-rt/sps-ci/NativeDylibManagerSPSCI.h"
#include "orc-rt/NativeDylibManager.h"
#include "orc-rt/SPSWrapperFunction.h"

namespace orc_rt::sps_ci {

ORC_RT_SPS_WRAPPER(
    orc_rt_ci_sps_NativeDylibManager_load,
    SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::load))

ORC_RT_SPS_WRAPPER(
    orc_rt_ci_sps_NativeDylibManager_unload,
    SPSError(SPSExecutorAddr, SPSExecutorAddr),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::unload))

ORC_RT_SPS_WRAPPER(
    orc_rt_ci_sps_NativeDylibManager_lookup,
    SPSExpected<SPSSequence<SPSExecutorAddr>>(SPSExecutorAddr, SPSExecutorAddr,
                                              SPSSequence<SPSString>),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::lookup))

static std::pair<const char *, const void *>
    orc_rt_ci_NativeDylibManager_sps_interface[] = {
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_NativeDylibManager_load),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_NativeDylibManager_unload),
        ORC_RT_SYMTAB_PAIR(orc_rt_ci_sps_NativeDylibManager_lookup)};

Error addNativeDylibManager(SimpleSymbolTable &ST) {
  return ST.addUnique(orc_rt_ci_NativeDylibManager_sps_interface);
}

} // namespace orc_rt::sps_ci
