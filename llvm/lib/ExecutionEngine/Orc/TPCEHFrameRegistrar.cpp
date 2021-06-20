//===------ TPCEHFrameRegistrar.cpp - TPC-based eh-frame registration -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TPCEHFrameRegistrar.h"
#include "llvm/Support/BinaryStreamWriter.h"

using namespace llvm::orc::shared;

namespace llvm {
namespace orc {

Expected<std::unique_ptr<TPCEHFrameRegistrar>>
TPCEHFrameRegistrar::Create(TargetProcessControl &TPC) {
  // FIXME: Proper mangling here -- we really need to decouple linker mangling
  // from DataLayout.

  // Find the addresses of the registration/deregistration functions in the
  // target process.
  auto ProcessHandle = TPC.loadDylib(nullptr);
  if (!ProcessHandle)
    return ProcessHandle.takeError();

  std::string RegisterWrapperName, DeregisterWrapperName;
  if (TPC.getTargetTriple().isOSBinFormatMachO()) {
    RegisterWrapperName += '_';
    DeregisterWrapperName += '_';
  }
  RegisterWrapperName += "llvm_orc_registerEHFrameSectionWrapper";
  DeregisterWrapperName += "llvm_orc_deregisterEHFrameSectionWrapper";

  SymbolLookupSet RegistrationSymbols;
  RegistrationSymbols.add(TPC.intern(RegisterWrapperName));
  RegistrationSymbols.add(TPC.intern(DeregisterWrapperName));

  auto Result = TPC.lookupSymbols({{*ProcessHandle, RegistrationSymbols}});
  if (!Result)
    return Result.takeError();

  assert(Result->size() == 1 && "Unexpected number of dylibs in result");
  assert((*Result)[0].size() == 2 &&
         "Unexpected number of addresses in result");

  auto RegisterEHFrameWrapperFnAddr = (*Result)[0][0];
  auto DeregisterEHFrameWrapperFnAddr = (*Result)[0][1];

  return std::make_unique<TPCEHFrameRegistrar>(
      TPC, RegisterEHFrameWrapperFnAddr, DeregisterEHFrameWrapperFnAddr);
}

Error TPCEHFrameRegistrar::registerEHFrames(JITTargetAddress EHFrameSectionAddr,
                                            size_t EHFrameSectionSize) {

  return WrapperFunction<void(SPSTargetAddress, uint64_t)>::call(
      TPCCaller(TPC, RegisterEHFrameWrapperFnAddr), EHFrameSectionAddr,
      static_cast<uint64_t>(EHFrameSectionSize));
}

Error TPCEHFrameRegistrar::deregisterEHFrames(
    JITTargetAddress EHFrameSectionAddr, size_t EHFrameSectionSize) {
  return WrapperFunction<void(SPSTargetAddress, uint64_t)>::call(
      TPCCaller(TPC, DeregisterEHFrameWrapperFnAddr), EHFrameSectionAddr,
      static_cast<uint64_t>(EHFrameSectionSize));
}

} // end namespace orc
} // end namespace llvm
