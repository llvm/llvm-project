//===----- EPCDebugObjectRegistrar.cpp - EPC-based debug registration -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/Support/BinaryStreamWriter.h"

namespace llvm {
namespace orc {

Expected<std::unique_ptr<EPCDebugObjectRegistrar>>
createJITLoaderGDBRegistrar(ExecutionSession &ES) {
  auto &EPC = ES.getExecutorProcessControl();
  auto ProcessHandle = EPC.loadDylib(nullptr);
  if (!ProcessHandle)
    return ProcessHandle.takeError();

  SymbolStringPtr RegisterFn =
      EPC.getTargetTriple().isOSBinFormatMachO()
          ? EPC.intern("_llvm_orc_registerJITLoaderGDBWrapper")
          : EPC.intern("llvm_orc_registerJITLoaderGDBWrapper");

  SymbolLookupSet RegistrationSymbols;
  RegistrationSymbols.add(RegisterFn);

  auto Result = EPC.lookupSymbols({{*ProcessHandle, RegistrationSymbols}});
  if (!Result)
    return Result.takeError();

  assert(Result->size() == 1 && "Unexpected number of dylibs in result");
  assert((*Result)[0].size() == 1 &&
         "Unexpected number of addresses in result");

  return std::make_unique<EPCDebugObjectRegistrar>(
      ES, ExecutorAddr((*Result)[0][0]));
}

Error EPCDebugObjectRegistrar::registerDebugObject(
    ExecutorAddrRange TargetMem) {
  return ES.callSPSWrapper<void(shared::SPSExecutorAddrRange)>(RegisterFn,
                                                               TargetMem);
}

} // namespace orc
} // namespace llvm
