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
  // Instance is the executor-side manager object -- a data symbol passed as the
  // first argument to each call, not a wrapper to proxy. The proxies resolve to
  // the specs' default (NativeDylibManager) names.
  if (auto Err = lookupAndApply(
          JD,
          {recordAddr(rt::sps_ci::NativeDylibManagerInstanceName, &B.Instance),
           recordProxy<DylibMgrOpenProxySpec>(&B.Open),
           recordProxy<DylibMgrResolveProxySpec>(&B.Resolve)}))
    return std::move(Err);
  return std::make_unique<EPCGenericDylibManager>(ES, std::move(B));
}

Expected<std::unique_ptr<EPCGenericDylibManager>>
createEPCGenericDylibManager(ExecutionSession &ES) {
  return createEPCGenericDylibManager(ES.getBootstrapJITDylib());
}

} // namespace llvm::orc::sps
