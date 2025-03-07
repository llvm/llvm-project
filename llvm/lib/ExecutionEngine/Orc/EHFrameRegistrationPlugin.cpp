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

#define DEBUG_TYPE "orc"

using namespace llvm::jitlink;

namespace llvm::orc {

EHFrameRegistrationPlugin::EHFrameRegistrationPlugin(
    ExecutionSession &ES, std::unique_ptr<EHFrameRegistrar> Registrar)
    : ES(ES), Registrar(std::move(Registrar)) {}

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

  PassConfig.PostFixupPasses.push_back(createEHFrameRecorderPass(
      LG.getTargetTriple(), [this, &MR](ExecutorAddr Addr, size_t Size) {
        if (Addr) {
          std::lock_guard<std::mutex> Lock(EHFramePluginMutex);
          assert(!InProcessLinks.count(&MR) &&
                 "Link for MR already being tracked?");
          InProcessLinks[&MR] = {Addr, Size};
        }
      }));
}

Error EHFrameRegistrationPlugin::notifyEmitted(
    MaterializationResponsibility &MR) {

  ExecutorAddrRange EmittedRange;
  {
    std::lock_guard<std::mutex> Lock(EHFramePluginMutex);

    auto EHFrameRangeItr = InProcessLinks.find(&MR);
    if (EHFrameRangeItr == InProcessLinks.end())
      return Error::success();

    EmittedRange = EHFrameRangeItr->second;
    assert(EmittedRange.Start && "eh-frame addr to register can not be null");
    InProcessLinks.erase(EHFrameRangeItr);
  }

  if (auto Err = MR.withResourceKeyDo(
          [&](ResourceKey K) { EHFrameRanges[K].push_back(EmittedRange); }))
    return Err;

  return Registrar->registerEHFrames(EmittedRange);
}

Error EHFrameRegistrationPlugin::notifyFailed(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(EHFramePluginMutex);
  InProcessLinks.erase(&MR);
  return Error::success();
}

Error EHFrameRegistrationPlugin::notifyRemovingResources(JITDylib &JD,
                                                         ResourceKey K) {
  std::vector<ExecutorAddrRange> RangesToRemove;

  ES.runSessionLocked([&] {
    auto I = EHFrameRanges.find(K);
    if (I != EHFrameRanges.end()) {
      RangesToRemove = std::move(I->second);
      EHFrameRanges.erase(I);
    }
  });

  Error Err = Error::success();
  while (!RangesToRemove.empty()) {
    auto RangeToRemove = RangesToRemove.back();
    RangesToRemove.pop_back();
    assert(RangeToRemove.Start && "Untracked eh-frame range must not be null");
    Err = joinErrors(std::move(Err),
                     Registrar->deregisterEHFrames(RangeToRemove));
  }

  return Err;
}

void EHFrameRegistrationPlugin::notifyTransferringResources(
    JITDylib &JD, ResourceKey DstKey, ResourceKey SrcKey) {
  auto SI = EHFrameRanges.find(SrcKey);
  if (SI == EHFrameRanges.end())
    return;

  auto DI = EHFrameRanges.find(DstKey);
  if (DI != EHFrameRanges.end()) {
    auto &SrcRanges = SI->second;
    auto &DstRanges = DI->second;
    DstRanges.reserve(DstRanges.size() + SrcRanges.size());
    for (auto &SrcRange : SrcRanges)
      DstRanges.push_back(std::move(SrcRange));
    EHFrameRanges.erase(SI);
  } else {
    // We need to move SrcKey's ranges over without invalidating the SI
    // iterator.
    auto Tmp = std::move(SI->second);
    EHFrameRanges.erase(SI);
    EHFrameRanges[DstKey] = std::move(Tmp);
  }
}

} // namespace llvm::orc
