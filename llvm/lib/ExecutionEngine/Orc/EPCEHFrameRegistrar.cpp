//===------ EPCEHFrameRegistrar.cpp - EPC-based eh-frame registration -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCEHFrameRegistrar.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/BinaryStreamWriter.h"

using namespace llvm::orc::shared;

namespace llvm {
namespace orc {

Expected<std::unique_ptr<EPCEHFrameRegistrar>> EPCEHFrameRegistrar::Create(
    ExecutionSession &ES,
    std::optional<ExecutorAddr> RegistrationFunctionsDylib) {
  // FIXME: Proper mangling here -- we really need to decouple linker mangling
  // from DataLayout.

  // Find the addresses of the registration/deregistration functions in the
  // executor process.
  auto &EPC = ES.getExecutorProcessControl();

  if (!RegistrationFunctionsDylib) {
    if (auto D = EPC.loadDylib(nullptr))
      RegistrationFunctionsDylib = *D;
    else
      return D.takeError();
  }

  std::string RegisterWrapperName, DeregisterWrapperName;
  if (EPC.getTargetTriple().isOSBinFormatMachO()) {
    RegisterWrapperName += '_';
    DeregisterWrapperName += '_';
  }
  RegisterWrapperName += "llvm_orc_registerEHFrameSectionWrapper";
  DeregisterWrapperName += "llvm_orc_deregisterEHFrameSectionWrapper";

  SymbolLookupSet RegistrationSymbols;
  RegistrationSymbols.add(EPC.intern(RegisterWrapperName));
  RegistrationSymbols.add(EPC.intern(DeregisterWrapperName));

  auto Result =
      EPC.lookupSymbols({{*RegistrationFunctionsDylib, RegistrationSymbols}});
  if (!Result)
    return Result.takeError();

  assert(Result->size() == 1 && "Unexpected number of dylibs in result");
  assert((*Result)[0].size() == 2 &&
         "Unexpected number of addresses in result");

  auto RegisterEHFrameWrapperFnAddr = (*Result)[0][0];
  auto DeregisterEHFrameWrapperFnAddr = (*Result)[0][1];

  return std::make_unique<EPCEHFrameRegistrar>(ES, RegisterEHFrameWrapperFnAddr,
                                               DeregisterEHFrameWrapperFnAddr);
}

Error EPCEHFrameRegistrar::registerEHFrames(ExecutorAddrRange EHFrameSection) {
  return ES.callSPSWrapper<void(SPSExecutorAddrRange)>(
      RegisterEHFrameWrapperFnAddr, EHFrameSection);
}

Error EPCEHFrameRegistrar::deregisterEHFrames(
    ExecutorAddrRange EHFrameSection) {
  return ES.callSPSWrapper<void(SPSExecutorAddrRange)>(
      DeregisterEHFrameWrapperFnAddr, EHFrameSection);
}

} // end namespace orc
} // end namespace llvm
