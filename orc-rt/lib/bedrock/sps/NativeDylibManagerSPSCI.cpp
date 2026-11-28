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
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/support/sps/SPSSymbolLookupResult.h"
#include "orc-rt/support/sps/SPSSymbolLookupSet.h"

namespace orc_rt::sps::ci {

// Unlike the operations below, create takes no instance address: the new
// instance is owned by the calling Session, which is passed to every wrapper.
extern "C" ORC_RT_SPS_WRAPPER_SIG(orc_rt_ci_sps_NativeDylibManager_create) {
  assert(S && "Session pointer must not be null");
  SPSWrapperFunction<SPSExpected<SPSExecutorAddr>()>::handle(
      S, ArgBytes, Return, CallId,
      [S](move_only_function<void(Expected<void *>)> SendResult) {
        // Calls from the controller hold a keepalive, so the Session cannot
        // have started shutting down its services yet: the new instance will
        // be notified of detach and shutdown like any other service.
        auto &Ses = *unwrap(S);
        auto &NDM = Ses.createService<NativeDylibManager>(Ses);
        SendResult(static_cast<void *>(&NDM));
      });
}

ORC_RT_SPS_WRAPPER_IMPL(
    orc_rt_ci_sps_NativeDylibManager_load,
    SPSExpected<SPSExecutorAddr>(SPSExecutorAddr, SPSString),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::load))

ORC_RT_SPS_WRAPPER_IMPL(
    orc_rt_ci_sps_NativeDylibManager_lookup,
    SPSExpected<SPSSymbolLookupResult>(SPSExecutorAddr, SPSExecutorAddr,
                                       SPSSymbolLookupSet),
    WrapperFunction::handleWithAsyncMethod(&NativeDylibManager::lookup))

static std::pair<SymbolNameSpec, const void *>
    orc_rt_ci_NativeDylibManager_sps_interface[] = {
        ORC_RT_SYMTAB_C_PAIR(orc_rt_ci_sps_NativeDylibManager_create),
        ORC_RT_SYMTAB_C_PAIR(orc_rt_ci_sps_NativeDylibManager_load),
        ORC_RT_SYMTAB_C_PAIR(orc_rt_ci_sps_NativeDylibManager_lookup)};

Error addNativeDylibManager(SimpleSymbolTable &ST) {
  return ST.addUnique(orc_rt_ci_NativeDylibManager_sps_interface);
}

} // namespace orc_rt::sps::ci
