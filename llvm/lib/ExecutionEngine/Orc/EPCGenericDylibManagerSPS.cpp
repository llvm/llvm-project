//===- EPCGenericDylibManagerSPS.cpp - SPS dylib management ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericDylibManagerSPS.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LookupAndApply.h"
#include "llvm/ExecutionEngine/Orc/RecordProxy.h"

namespace llvm::orc::sps {

Expected<std::unique_ptr<EPCGenericDylibManager>>
createEPCGenericDylibManager(JITDylib &JD) {
  auto &ES = JD.getExecutionSession();
  EPCGenericDylibManager::Bindings B;
  Proxy<Expected<ExecutorAddr>()> Create;

  // The instance -- the executor-side manager object passed as the first
  // argument to each call -- is either created on demand via Create, or (for
  // executors that do not provide Create) published as a data symbol. Both are
  // looked up weakly; whichever is available is used below. The operation
  // proxies resolve to the specs' default (NativeDylibManager) names.
  if (auto Err = lookupAndApply(
          JD,
          {recordProxy<DylibMgrCreateProxySpec>(
               &Create, SymbolLookupFlags::WeaklyReferencedSymbol),
           recordAddr(rt::sps_ci::NativeDylibManagerInstanceName, &B.Instance,
                      SymbolLookupFlags::WeaklyReferencedSymbol),
           recordProxy<DylibMgrOpenProxySpec>(&B.Open),
           recordProxy<DylibMgrResolveProxySpec>(&B.Resolve)}))
    return std::move(Err);

  if (Create) {
    auto Instance = Create(ES);
    if (!Instance)
      return Instance.takeError();
    B.Instance = *Instance;
  } else if (!B.Instance)
    return make_error<StringError>(
        "Executor provides neither " + DylibMgrCreateProxySpec::Name.getName() +
            " nor " + rt::sps_ci::NativeDylibManagerInstanceName.getName() +
            ": cannot bind a NativeDylibManager instance",
        inconvertibleErrorCode());

  return std::make_unique<EPCGenericDylibManager>(ES, std::move(B));
}

Expected<std::unique_ptr<EPCGenericDylibManager>>
createEPCGenericDylibManager(ExecutionSession &ES) {
  return createEPCGenericDylibManager(ES.getBootstrapJITDylib());
}

} // namespace llvm::orc::sps
