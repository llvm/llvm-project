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
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"

namespace llvm {
namespace orc {
namespace shared {

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

Expected<EPCGenericDylibManager>
EPCGenericDylibManager::CreateWithDefaultBootstrapSymbols(
    ExecutorProcessControl &EPC) {
  SymbolAddrs SAs;
  if (auto Err = EPC.getBootstrapSymbols(
          {{SAs.Instance, rt::SimpleExecutorDylibManagerInstanceName},
           {SAs.Open, rt::SimpleExecutorDylibManagerOpenWrapperName},
           {SAs.Resolve, rt::SimpleExecutorDylibManagerResolveWrapperName}}))
    return std::move(Err);
  return EPCGenericDylibManager(EPC, std::move(SAs));
}

Expected<EPCGenericDylibManager>
EPCGenericDylibManager::Create(JITDylib &JD,
                               rt::SimpleExecutorDylibManagerSymbolNames SNs) {
  auto &ES = JD.getExecutionSession();
  SymbolAddrs SAs;
  if (auto Err = lookupAndRecordAddrs(
          ES, LookupKind::Static, makeJITDylibSearchOrder({&JD}),
          {
              {ES.intern(SNs.InstanceName), &SAs.Instance},
              {ES.intern(SNs.OpenName), &SAs.Open},
              {ES.intern(SNs.ResolveName), &SAs.Resolve},
          }))
    return std::move(Err);
  return EPCGenericDylibManager(ES.getExecutorProcessControl(), std::move(SAs));
}

Expected<EPCGenericDylibManager>
EPCGenericDylibManager::Create(ExecutionSession &ES,
                               rt::SimpleExecutorDylibManagerSymbolNames SNs) {
  return Create(ES.getBootstrapJITDylib(), std::move(SNs));
}

Expected<tpctypes::DylibHandle> EPCGenericDylibManager::open(StringRef Path,
                                                             uint64_t Mode) {
  Expected<tpctypes::DylibHandle> H((ExecutorAddr()));
  if (auto Err =
          EPC.callSPSWrapper<rt::SPSSimpleExecutorDylibManagerOpenSignature>(
              SAs.Open, H, SAs.Instance, Path, Mode))
    return std::move(Err);
  return H;
}

void EPCGenericDylibManager::lookupAsync(tpctypes::DylibHandle H,
                                         const SymbolLookupSet &Lookup,
                                         SymbolLookupCompleteFn Complete) {
  EPC.callSPSWrapperAsync<rt::SPSSimpleExecutorDylibManagerResolveSignature>(
      SAs.Resolve,
      [Complete = std::move(Complete)](
          Error SerializationErr,
          Expected<std::vector<std::optional<ExecutorAddr>>> Result) mutable {
        if (SerializationErr) {
          cantFail(Result.takeError());
          Complete(std::move(SerializationErr));
          return;
        }
        Complete(std::move(Result));
      },
      SAs.Instance, H, Lookup);
}

void EPCGenericDylibManager::lookupAsync(tpctypes::DylibHandle H,
                                         const RemoteSymbolLookupSet &Lookup,
                                         SymbolLookupCompleteFn Complete) {
  EPC.callSPSWrapperAsync<rt::SPSSimpleExecutorDylibManagerResolveSignature>(
      SAs.Resolve,
      [Complete = std::move(Complete)](
          Error SerializationErr,
          Expected<std::vector<std::optional<ExecutorAddr>>> Result) mutable {
        if (SerializationErr) {
          cantFail(Result.takeError());
          Complete(std::move(SerializationErr));
          return;
        }
        Complete(std::move(Result));
      },
      SAs.Instance, H, Lookup);
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
