//===- SimpleMemoryMapSPS.cpp - SPS memory-map bindings -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SimpleMemoryMapSPS.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RecordProxy.h"

namespace llvm::orc::sps {

Expected<SimpleMemoryMapBindings> createSimpleMemoryMapBindings(JITDylib &JD) {
  SimpleMemoryMapBindings B;
  Proxy<Expected<ExecutorAddr>()> Create;

  // The instance -- the executor-side manager object passed as the first
  // argument to each call -- is either created on demand via Create, or (for
  // executors that do not provide Create) published as a data symbol. Both are
  // looked up weakly; whichever is available is used below. The operation
  // proxies resolve to the specs' default (SimpleNativeMemoryMap) names.
  if (auto Err = lookupAndApply(
          JD,
          {recordProxy<MemMgrCreateProxySpec>(
               &Create, SymbolLookupFlags::WeaklyReferencedSymbol),
           recordAddr(rt::sps_ci::SimpleNativeMemoryMapInstanceName,
                      &B.Instance, SymbolLookupFlags::WeaklyReferencedSymbol),
           recordProxy<MemMgrReserveProxySpec>(&B.Reserve),
           recordProxy<MemMgrInitializeProxySpec>(&B.Initialize),
           recordProxy<MemMgrDeinitializeProxySpec>(&B.Deinitialize),
           recordProxy<MemMgrReleaseProxySpec>(&B.Release)}))
    return std::move(Err);

  if (Create) {
    auto Instance = Create(JD.getExecutionSession());
    if (!Instance)
      return Instance.takeError();
    B.Instance = *Instance;
  } else if (!B.Instance)
    return make_error<StringError>(
        "Executor provides neither " + MemMgrCreateProxySpec::Name.getName() +
            " nor " + rt::sps_ci::SimpleNativeMemoryMapInstanceName.getName() +
            ": cannot bind a SimpleNativeMemoryMap instance",
        inconvertibleErrorCode());

  return std::move(B);
}

Expected<SimpleMemoryMapBindings>
createSimpleMemoryMapBindings(ExecutionSession &ES) {
  return createSimpleMemoryMapBindings(ES.getBootstrapJITDylib());
}

} // namespace llvm::orc::sps
