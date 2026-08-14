//===------- EPCGenericDylibManager.cpp -- Dylib management via EPC -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericDylibManager.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"
#include "llvm/ExecutionEngine/Orc/RTBridge/SPS/ProxySpec.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/NativeDylibManagerSPSCI.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"

namespace llvm {
namespace orc {
namespace shared {

// Serialize a local SymbolLookupSet directly as an SPSRemoteSymbolLookupSet,
// avoiding a std::string copy of each (potentially large) symbol name.
template <>
class SPSSerializationTraits<SPSRemoteSymbolLookupSetElement,
                             SymbolLookupSet::value_type> {
public:
  static size_t size(const SymbolLookupSet::value_type &V) {
    return SPSArgList<SPSString, bool>::size(
        *V.first, V.second == SymbolLookupFlags::RequiredSymbol);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const SymbolLookupSet::value_type &V) {
    return SPSArgList<SPSString, bool>::serialize(
        OB, *V.first, V.second == SymbolLookupFlags::RequiredSymbol);
  }
};

template <>
class TrivialSPSSequenceSerialization<SPSRemoteSymbolLookupSetElement,
                                      SymbolLookupSet> {
public:
  static constexpr bool available = true;
};

} // end namespace shared

namespace {

using OpenSpec = rt::sps::ProxySpec<EPCGenericDylibManager::OpenProxy,
                                    rt::sps_ci::DylibMgrOpen>;
using ResolveSpec = rt::sps::ProxySpec<EPCGenericDylibManager::ResolveProxy,
                                       rt::sps_ci::DylibMgrResolve>;

} // namespace

Expected<EPCGenericDylibManager> EPCGenericDylibManager::Create(JITDylib &JD) {
  auto &ES = JD.getExecutionSession();
  Bindings B;
  // Instance is the executor-side manager object -- a data symbol passed as the
  // first argument to each call, not a wrapper to proxy.
  if (auto Err = lookupAndRecordAddrs(
          ES, LookupKind::Static, makeJITDylibSearchOrder({&JD}),
          {{ES.intern(rt::sps_ci::NativeDylibManagerInstanceName),
            &B.Instance}}))
    return std::move(Err);
  // The proxies resolve to the specs' default (NativeDylibManager) names.
  if (auto Err = rt::buildProxies(JD, rt::proxyInit<OpenSpec>(&B.Open),
                                  rt::proxyInit<ResolveSpec>(&B.Resolve)))
    return std::move(Err);
  return EPCGenericDylibManager(ES, std::move(B));
}

Expected<EPCGenericDylibManager>
EPCGenericDylibManager::Create(ExecutionSession &ES) {
  return Create(ES.getBootstrapJITDylib());
}

Expected<tpctypes::DylibHandle> EPCGenericDylibManager::open(StringRef Path,
                                                             uint64_t Mode) {
  return B.Open(ES, B.Instance, Path, Mode);
}

void EPCGenericDylibManager::lookupAsync(tpctypes::DylibHandle H,
                                         const SymbolLookupSet &Lookup,
                                         SymbolLookupCompleteFn Complete) {
  B.Resolve(std::move(Complete), ES, B.Instance, H, Lookup);
}

Expected<tpctypes::DylibHandle>
EPCGenericDylibManager::loadDylib(const char *DylibPath) {
  return open(DylibPath, 0);
}

void EPCGenericDylibManager::lookupSymbolsAsync(
    tpctypes::DylibHandle H, const SymbolLookupSet &Symbols,
    DylibManager::SymbolLookupCompleteFn Complete) {
  lookupAsync(H, Symbols, std::move(Complete));
}

} // end namespace orc
} // end namespace llvm
