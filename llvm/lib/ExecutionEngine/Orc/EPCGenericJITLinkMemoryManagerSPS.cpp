//===- EPCGenericJITLinkMemoryManagerSPS.cpp - SPS mem manager ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManagerSPS.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"

namespace llvm::orc::sps {

Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
createEPCGenericJITLinkMemoryManager(JITDylib &JD) {
  auto &ES = JD.getExecutionSession();
  EPCGenericJITLinkMemoryManager::Bindings B;
  // Instance is the executor-side allocator object -- a data symbol passed as
  // the first argument to each call, not a wrapper to proxy.
  if (auto Err = lookupAndRecordAddrs(
          ES, LookupKind::Static, makeJITDylibSearchOrder({&JD}),
          {{ES.intern(rt::sps_ci::SimpleNativeMemoryMapInstanceName),
            &B.Instance}}))
    return std::move(Err);
  // The proxies resolve to the specs' default (SimpleNativeMemoryMap) names.
  if (auto Err =
          buildProxies(JD, proxyInit<MemMgrReserveProxySpec>(&B.Reserve),
                       proxyInit<MemMgrInitializeProxySpec>(&B.Initialize),
                       proxyInit<MemMgrDeinitializeProxySpec>(&B.Deinitialize),
                       proxyInit<MemMgrReleaseProxySpec>(&B.Release)))
    return std::move(Err);
  return std::make_unique<EPCGenericJITLinkMemoryManager>(ES, std::move(B));
}

Expected<std::unique_ptr<EPCGenericJITLinkMemoryManager>>
createEPCGenericJITLinkMemoryManager(ExecutionSession &ES) {
  return createEPCGenericJITLinkMemoryManager(ES.getBootstrapJITDylib());
}

} // namespace llvm::orc::sps
