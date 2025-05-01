//===----- UnwindInfoRegistrationPlugin.cpp - libunwind registration ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/UnwindInfoRegistrationPlugin.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ExecutionEngine/Orc/Shared/MachOObjectFormat.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "orc"

using namespace llvm::jitlink;

namespace llvm::orc {

Expected<std::shared_ptr<UnwindInfoRegistrationPlugin>>
UnwindInfoRegistrationPlugin::Create(ExecutionSession &ES) {

  ExecutorAddr Register, Deregister;

  auto &EPC = ES.getExecutorProcessControl();
  if (auto Err = EPC.getBootstrapSymbols(
          {{Register, rt_alt::UnwindInfoManagerRegisterActionName},
           {Deregister, rt_alt::UnwindInfoManagerDeregisterActionName}}))
    return std::move(Err);

  return std::make_shared<UnwindInfoRegistrationPlugin>(ES, Register,
                                                        Deregister);
}

void UnwindInfoRegistrationPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, LinkGraph &G,
    PassConfiguration &PassConfig) {

  PassConfig.PostFixupPasses.push_back(
      [this](LinkGraph &G) { return addUnwindInfoRegistrationActions(G); });
}

Error UnwindInfoRegistrationPlugin::addUnwindInfoRegistrationActions(
    LinkGraph &G) {
  ExecutorAddrRange EHFrameRange, UnwindInfoRange;

  std::vector<Block *> CodeBlocks;

  auto ScanUnwindInfoSection = [&](Section &Sec, ExecutorAddrRange &SecRange) {
    if (Sec.empty())
      return;

    SecRange.Start = (*Sec.blocks().begin())->getAddress();
    for (auto *B : Sec.blocks()) {
      auto R = B->getRange();
      SecRange.Start = std::min(SecRange.Start, R.Start);
      SecRange.End = std::max(SecRange.End, R.End);
      for (auto &E : B->edges()) {
        if (E.getKind() != Edge::KeepAlive || !E.getTarget().isDefined())
          continue;
        auto &TargetBlock = E.getTarget().getBlock();
        auto &TargetSection = TargetBlock.getSection();
        if ((TargetSection.getMemProt() & MemProt::Exec) == MemProt::Exec)
          CodeBlocks.push_back(&TargetBlock);
      }
    }
  };

  if (auto *EHFrame = G.findSectionByName(MachOEHFrameSectionName))
    ScanUnwindInfoSection(*EHFrame, EHFrameRange);

  if (auto *UnwindInfo = G.findSectionByName(MachOUnwindInfoSectionName))
    ScanUnwindInfoSection(*UnwindInfo, UnwindInfoRange);

  if (CodeBlocks.empty())
    return Error::success();

  if ((EHFrameRange == ExecutorAddrRange() &&
       UnwindInfoRange == ExecutorAddrRange()))
    return Error::success();

  llvm::sort(CodeBlocks, [](const Block *LHS, const Block *RHS) {
    return LHS->getAddress() < RHS->getAddress();
  });

  SmallVector<ExecutorAddrRange> CodeRanges;
  for (auto *B : CodeBlocks) {
    if (CodeRanges.empty() || CodeRanges.back().End != B->getAddress())
      CodeRanges.push_back(B->getRange());
    else
      CodeRanges.back().End = B->getRange().End;
  }

  ExecutorAddr DSOBase;
  if (auto *DSOBaseSym = G.findAbsoluteSymbolByName(DSOBaseName))
    DSOBase = DSOBaseSym->getAddress();
  else if (auto *DSOBaseSym = G.findExternalSymbolByName(DSOBaseName))
    DSOBase = DSOBaseSym->getAddress();
  else if (auto *DSOBaseSym = G.findDefinedSymbolByName(DSOBaseName))
    DSOBase = DSOBaseSym->getAddress();
  else
    return make_error<StringError>("In " + G.getName() +
                                       " could not find dso base symbol",
                                   inconvertibleErrorCode());

  using namespace shared;
  using SPSRegisterArgs =
      SPSArgList<SPSSequence<SPSExecutorAddrRange>, SPSExecutorAddr,
                 SPSExecutorAddrRange, SPSExecutorAddrRange>;
  using SPSDeregisterArgs = SPSArgList<SPSSequence<SPSExecutorAddrRange>>;

  G.allocActions().push_back(
      {cantFail(WrapperFunctionCall::Create<SPSRegisterArgs>(
           Register, CodeRanges, DSOBase, EHFrameRange, UnwindInfoRange)),
       cantFail(WrapperFunctionCall::Create<SPSDeregisterArgs>(Deregister,
                                                               CodeRanges))});

  return Error::success();
}

} // namespace llvm::orc
