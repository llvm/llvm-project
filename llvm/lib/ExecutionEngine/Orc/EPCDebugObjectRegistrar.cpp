//===----- EPCDebugObjectRegistrar.cpp - EPC-based debug registration -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"

#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

Expected<std::unique_ptr<EPCDebugObjectRegistrar>> createJITLoaderGDBRegistrar(
    ExecutionSession &ES,
    std::optional<ExecutorAddr> RegistrationFunctionDylib) {
  auto &EPC = ES.getExecutorProcessControl();

  if (!RegistrationFunctionDylib) {
    if (auto D = EPC.getDylibMgr().loadDylib(nullptr))
      RegistrationFunctionDylib = *D;
    else
      return D.takeError();
  }

  SymbolStringPtr RegisterFn =
      EPC.getTargetTriple().isOSBinFormatMachO()
          ? EPC.intern("_llvm_orc_registerJITLoaderGDBWrapper")
          : EPC.intern("llvm_orc_registerJITLoaderGDBWrapper");

  SymbolLookupSet RegistrationSymbols;
  RegistrationSymbols.add(RegisterFn);

  auto Result = EPC.getDylibMgr().lookupSymbols(
      {{*RegistrationFunctionDylib, RegistrationSymbols}});
  if (!Result)
    return Result.takeError();

  assert(Result->size() == 1 && "Unexpected number of dylibs in result");
  assert((*Result)[0].size() == 1 &&
         "Unexpected number of addresses in result");

  if (!(*Result)[0][0].has_value())
    return make_error<StringError>(
        "Expected a valid address in the lookup result",
        inconvertibleErrorCode());

  ExecutorAddr RegisterAddr = (*Result)[0][0]->getAddress();
  return std::make_unique<EPCDebugObjectRegistrar>(ES, RegisterAddr);
}

Error EPCDebugObjectRegistrar::registerDebugObject(ExecutorAddrRange TargetMem,
                                                   bool AutoRegisterCode) {
  return ES.callSPSWrapper<void(shared::SPSExecutorAddrRange, bool)>(
      RegisterFn, TargetMem, AutoRegisterCode);
}

} // namespace orc
} // namespace llvm
