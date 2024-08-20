//===- CtxProfAnalysis.cpp - contextual profile analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the contextual profile analysis, which maintains contextual
// profiling info through IPO passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CtxProfAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "ctx_prof"

using namespace llvm;
cl::opt<std::string>
    UseCtxProfile("use-ctx-profile", cl::init(""), cl::Hidden,
                  cl::desc("Use the specified contextual profile file"));

namespace llvm {
namespace json {
Value toJSON(const PGOCtxProfContext &P) {
  Object Ret;
  Ret["Guid"] = P.guid();
  Ret["Counters"] = Array(P.counters());
  if (P.callsites().empty())
    return Ret;
  auto AllCS =
      ::llvm::map_range(P.callsites(), [](const auto &P) { return P.first; });
  auto MaxIt = ::llvm::max_element(AllCS);
  assert(MaxIt != AllCS.end() && "We should have a max value because the "
                                 "callsites collection is not empty.");
  Array CSites;
  // Iterate to, and including, the maximum index.
  for (auto I = 0U, Max = *MaxIt; I <= Max; ++I) {
    CSites.push_back(Array());
    Array &Targets = *CSites.back().getAsArray();
    if (P.hasCallsite(I))
      for (const auto &[_, Ctx] : P.callsite(I))
        Targets.push_back(toJSON(Ctx));
  }
  Ret["Callsites"] = std::move(CSites);

  return Ret;
}

Value toJSON(const PGOCtxProfContext::CallTargetMapTy &P) {
  Array Ret;
  for (const auto &[_, Ctx] : P)
    Ret.push_back(toJSON(Ctx));
  return Ret;
}
} // namespace json
} // namespace llvm

const char *AssignGUIDPass::GUIDMetadataName = "guid";

PreservedAnalyses AssignGUIDPass::run(Module &M, ModuleAnalysisManager &MAM) {
  for (auto &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    if (F.getMetadata(GUIDMetadataName))
      continue;
    const GlobalValue::GUID GUID = F.getGUID();
    F.setMetadata(GUIDMetadataName,
                  MDNode::get(M.getContext(),
                              {ConstantAsMetadata::get(ConstantInt::get(
                                  Type::getInt64Ty(M.getContext()), GUID))}));
  }
  return PreservedAnalyses::none();
}

GlobalValue::GUID AssignGUIDPass::getGUID(const Function &F) {
  if (F.isDeclaration()) {
    assert(GlobalValue::isExternalLinkage(F.getLinkage()));
    return GlobalValue::getGUID(F.getGlobalIdentifier());
  }
  auto *MD = F.getMetadata(GUIDMetadataName);
  assert(MD && "guid not found for defined function");
  return cast<ConstantInt>(cast<ConstantAsMetadata>(MD->getOperand(0))
                               ->getValue()
                               ->stripPointerCasts())
      ->getZExtValue();
}
AnalysisKey CtxProfAnalysis::Key;

CtxProfAnalysis::CtxProfAnalysis(StringRef Profile)
    : Profile(Profile.empty() ? UseCtxProfile : Profile) {}

PGOContextualProfile CtxProfAnalysis::run(Module &M,
                                          ModuleAnalysisManager &MAM) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Profile);
  if (auto EC = MB.getError()) {
    M.getContext().emitError("could not open contextual profile file: " +
                             EC.message());
    return {};
  }
  PGOCtxProfileReader Reader(MB.get()->getBuffer());
  auto MaybeCtx = Reader.loadContexts();
  if (!MaybeCtx) {
    M.getContext().emitError("contextual profile file is invalid: " +
                             toString(MaybeCtx.takeError()));
    return {};
  }

  PGOContextualProfile Result;

  for (const auto &F : M) {
    if (F.isDeclaration())
      continue;
    auto GUID = AssignGUIDPass::getGUID(F);
    assert(GUID && "guid not found for defined function");
    const auto &Entry = F.begin();
    uint32_t MaxCounters = 0; // we expect at least a counter.
    for (const auto &I : *Entry)
      if (auto *C = dyn_cast<InstrProfIncrementInst>(&I)) {
        MaxCounters =
            static_cast<uint32_t>(C->getNumCounters()->getZExtValue());
        break;
      }
    if (!MaxCounters)
      continue;
    uint32_t MaxCallsites = 0;
    for (const auto &BB : F)
      for (const auto &I : BB)
        if (auto *C = dyn_cast<InstrProfCallsite>(&I)) {
          MaxCallsites =
              static_cast<uint32_t>(C->getNumCounters()->getZExtValue());
          break;
        }
    auto [It, Ins] = Result.FuncInfo.insert(
        {GUID, PGOContextualProfile::FunctionInfo(F.getName())});
    (void)Ins;
    assert(Ins);
    It->second.NextCallsiteIndex = MaxCallsites;
    It->second.NextCounterIndex = MaxCounters;
  }
  // If we made it this far, the Result is valid - which we mark by setting
  // .Profiles.
  // Trim first the roots that aren't in this module.
  DenseSet<GlobalValue::GUID> ProfiledGUIDs;
  for (auto &[RootGuid, _] : llvm::make_early_inc_range(*MaybeCtx))
    if (!Result.FuncInfo.contains(RootGuid))
      MaybeCtx->erase(RootGuid);
  Result.Profiles = std::move(*MaybeCtx);
  return Result;
}

GlobalValue::GUID
PGOContextualProfile::getDefinedFunctionGUID(const Function &F) const {
  if (auto It = FuncInfo.find(AssignGUIDPass::getGUID(F)); It != FuncInfo.end())
    return It->first;
  return 0;
}

PreservedAnalyses CtxProfAnalysisPrinterPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  CtxProfAnalysis::Result &C = MAM.getResult<CtxProfAnalysis>(M);
  if (!C) {
    M.getContext().emitError("Invalid CtxProfAnalysis");
    return PreservedAnalyses::all();
  }

  OS << "Function Info:\n";
  for (const auto &[Guid, FuncInfo] : C.FuncInfo)
    OS << Guid << " : " << FuncInfo.Name
       << ". MaxCounterID: " << FuncInfo.NextCounterIndex
       << ". MaxCallsiteID: " << FuncInfo.NextCallsiteIndex << "\n";

  const auto JSONed = ::llvm::json::toJSON(C.profiles());

  OS << "\nCurrent Profile:\n";
  OS << formatv("{0:2}", JSONed);
  OS << "\n";
  return PreservedAnalyses::all();
}

InstrProfCallsite *CtxProfAnalysis::getCallsiteInstrumentation(CallBase &CB) {
  while (auto *Prev = CB.getPrevNode())
    if (auto *IPC = dyn_cast<InstrProfCallsite>(Prev))
      return IPC;
  return nullptr;
}
