//===----- UnwindInfoRegistrationPlugin.cpp - libunwind registration ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/UnwindInfoRegistrationPlugin.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Shared/MachOObjectFormat.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "orc"

using namespace llvm::jitlink;

static const char *FindDynamicUnwindSectionsFunctionName =
    "_orc_rt_alt_find_dynamic_unwind_sections";

namespace llvm::orc {

Expected<std::shared_ptr<UnwindInfoRegistrationPlugin>>
UnwindInfoRegistrationPlugin::Create(IRLayer &IRL, JITDylib &PlatformJD,
                                     ExecutorAddr Instance,
                                     ExecutorAddr FindHelper,
                                     ExecutorAddr Enable, ExecutorAddr Disable,
                                     ExecutorAddr Register,
                                     ExecutorAddr Deregister) {

  auto &ES = IRL.getExecutionSession();

  // Build bouncer module.
  auto M = makeBouncerModule(ES);
  if (!M)
    return M.takeError();

  auto BouncerRT = PlatformJD.createResourceTracker();
  auto RemoveBouncerModule = make_scope_exit([&]() {
    if (auto Err = BouncerRT->remove())
      ES.reportError(std::move(Err));
  });

  if (auto Err = PlatformJD.define(absoluteSymbols(
          {{ES.intern(rt_alt::UnwindInfoManagerInstanceName),
            ExecutorSymbolDef(Instance, JITSymbolFlags())},
           {ES.intern(rt_alt::UnwindInfoManagerFindSectionsHelperName),
            ExecutorSymbolDef(FindHelper, JITSymbolFlags::Callable)}})))
    return std::move(Err);

  if (auto Err = IRL.add(BouncerRT, std::move(*M)))
    return Err;

  auto FindUnwindSections =
      ES.lookup({&PlatformJD}, FindDynamicUnwindSectionsFunctionName);
  if (!FindUnwindSections)
    return FindUnwindSections.takeError();

  using namespace shared;
  using SPSEnableSig = SPSError(SPSExecutorAddr, SPSExecutorAddr);
  Error CallErr = Error::success();
  if (auto Err = ES.callSPSWrapper<SPSEnableSig>(
          Enable, CallErr, Instance, FindUnwindSections->getAddress())) {
    consumeError(std::move(CallErr));
    return std::move(Err);
  }

  if (CallErr)
    return std::move(CallErr);

  RemoveBouncerModule.release();

  return std::shared_ptr<UnwindInfoRegistrationPlugin>(
      new UnwindInfoRegistrationPlugin(ES, Instance, Disable, Register,
                                       Deregister));
}

Expected<std::shared_ptr<UnwindInfoRegistrationPlugin>>
UnwindInfoRegistrationPlugin::Create(IRLayer &IRL, JITDylib &PlatformJD) {

  ExecutorAddr Instance, FindHelper, Enable, Disable, Register, Deregister;

  auto &EPC = IRL.getExecutionSession().getExecutorProcessControl();
  if (auto Err = EPC.getBootstrapSymbols(
          {{Instance, rt_alt::UnwindInfoManagerInstanceName},
           {FindHelper, rt_alt::UnwindInfoManagerFindSectionsHelperName},
           {Enable, rt_alt::UnwindInfoManagerEnableWrapperName},
           {Disable, rt_alt::UnwindInfoManagerDisableWrapperName},
           {Register, rt_alt::UnwindInfoManagerRegisterActionName},
           {Deregister, rt_alt::UnwindInfoManagerDeregisterActionName}}))
    return std::move(Err);

  return Create(IRL, PlatformJD, Instance, FindHelper, Enable, Disable,
                Register, Deregister);
}

UnwindInfoRegistrationPlugin::~UnwindInfoRegistrationPlugin() {
  using namespace shared;
  using SPSDisableSig = SPSError(SPSExecutorAddr);
  Error CallErr = Error::success();
  if (auto Err = ES.callSPSWrapper<SPSDisableSig>(Disable, CallErr, Instance)) {
    consumeError(std::move(CallErr));
    ES.reportError(std::move(Err));
  }
  if (CallErr)
    ES.reportError(std::move(CallErr));
}

void UnwindInfoRegistrationPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, LinkGraph &G,
    PassConfiguration &PassConfig) {

  PassConfig.PostFixupPasses.push_back(
      [this](LinkGraph &G) { return addUnwindInfoRegistrationActions(G); });
}

Expected<ThreadSafeModule>
UnwindInfoRegistrationPlugin::makeBouncerModule(ExecutionSession &ES) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("__libunwind_find_unwind_bouncer", *Ctx);
  M->setTargetTriple(ES.getTargetTriple().str());

  auto EscapeName = [](const char *N) { return std::string("\01") + N; };

  auto *PtrTy = PointerType::getUnqual(*Ctx);
  auto *OpaqueStructTy = StructType::create(*Ctx, "UnwindInfoMgr");
  auto *UnwindMgrInstance = new GlobalVariable(
      *M, OpaqueStructTy, true, GlobalValue::ExternalLinkage, nullptr,
      EscapeName(rt_alt::UnwindInfoManagerInstanceName));

  auto *Int64Ty = Type::getInt64Ty(*Ctx);
  auto *FindHelperTy = FunctionType::get(Int64Ty, {PtrTy, PtrTy, PtrTy}, false);
  auto *FindHelperFn = Function::Create(
      FindHelperTy, GlobalValue::ExternalLinkage,
      EscapeName(rt_alt::UnwindInfoManagerFindSectionsHelperName), *M);

  auto *FindFnTy = FunctionType::get(Int64Ty, {PtrTy, PtrTy}, false);
  auto *FindFn =
      Function::Create(FindFnTy, GlobalValue::ExternalLinkage,
                       EscapeName(FindDynamicUnwindSectionsFunctionName), *M);
  auto *EntryBlock = BasicBlock::Create(M->getContext(), StringRef(), FindFn);
  IRBuilder<> IB(EntryBlock);

  std::vector<Value *> FindHelperArgs;
  FindHelperArgs.push_back(UnwindMgrInstance);
  for (auto &Arg : FindFn->args())
    FindHelperArgs.push_back(&Arg);

  IB.CreateRet(IB.CreateCall(FindHelperFn, FindHelperArgs));

  return ThreadSafeModule(std::move(M), std::move(Ctx));
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
      SPSArgList<SPSExecutorAddr, SPSSequence<SPSExecutorAddrRange>,
                 SPSExecutorAddr, SPSExecutorAddrRange, SPSExecutorAddrRange>;
  using SPSDeregisterArgs =
      SPSArgList<SPSExecutorAddr, SPSSequence<SPSExecutorAddrRange>>;

  G.allocActions().push_back(
      {cantFail(WrapperFunctionCall::Create<SPSRegisterArgs>(
           Register, Instance, CodeRanges, DSOBase, EHFrameRange,
           UnwindInfoRange)),
       cantFail(WrapperFunctionCall::Create<SPSDeregisterArgs>(
           Deregister, Instance, CodeRanges))});

  return Error::success();
}

} // namespace llvm::orc
