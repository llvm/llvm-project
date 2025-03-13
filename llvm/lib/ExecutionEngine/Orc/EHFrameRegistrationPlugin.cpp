//===--------- EHFrameRegistrationPlugin.cpp - Register eh-frames ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h"

#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"
#include "llvm/ExecutionEngine/Orc/Shared/MachOObjectFormat.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"

#define DEBUG_TYPE "orc"

using namespace llvm::jitlink;

namespace llvm::orc {

Expected<std::unique_ptr<EHFrameRegistrationPlugin>>
EHFrameRegistrationPlugin::Create(ExecutionSession &ES) {
  // Lookup addresseses of the registration/deregistration functions in the
  // bootstrap map.
  ExecutorAddr RegisterEHFrameSectionAllocAction;
  ExecutorAddr DeregisterEHFrameSectionAllocAction;
  if (auto Err = ES.getExecutorProcessControl().getBootstrapSymbols(
          {{RegisterEHFrameSectionAllocAction,
            rt::RegisterEHFrameSectionAllocActionName},
           {DeregisterEHFrameSectionAllocAction,
            rt::DeregisterEHFrameSectionAllocActionName}}))
    return std::move(Err);

  return std::make_unique<EHFrameRegistrationPlugin>(
      RegisterEHFrameSectionAllocAction, DeregisterEHFrameSectionAllocAction);
}

void EHFrameRegistrationPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, LinkGraph &LG,
    PassConfiguration &PassConfig) {
  if (LG.getTargetTriple().isOSBinFormatMachO())
    PassConfig.PrePrunePasses.insert(
        PassConfig.PrePrunePasses.begin(), [](LinkGraph &G) {
          if (auto *CUSec = G.findSectionByName(MachOCompactUnwindSectionName))
            G.removeSection(*CUSec);
          return Error::success();
        });

  PassConfig.PostFixupPasses.push_back([this](LinkGraph &G) -> Error {
    if (auto *EHFrame = getEHFrameSection(G)) {
      using namespace shared;
      auto R = SectionRange(*EHFrame).getRange();
      G.allocActions().push_back(
          {cantFail(
               WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange>>(
                   RegisterEHFrame, R)),
           cantFail(
               WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange>>(
                   DeregisterEHFrame, R))});
    }
    return Error::success();
  });
}

} // namespace llvm::orc
