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

static cl::opt<CtxProfAnalysisPrinterPass::PrintMode> PrintLevel(
    "ctx-profile-printer-level",
    cl::init(CtxProfAnalysisPrinterPass::PrintMode::JSON), cl::Hidden,
    cl::values(clEnumValN(CtxProfAnalysisPrinterPass::PrintMode::Everything,
                          "everything", "print everything - most verbose"),
               clEnumValN(CtxProfAnalysisPrinterPass::PrintMode::JSON, "json",
                          "just the json representation of the profile")),
    cl::desc("Verbosity level of the contextual profile printer pass."));

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

CtxProfAnalysis::CtxProfAnalysis(std::optional<StringRef> Profile)
    : Profile([&]() -> std::optional<StringRef> {
        if (Profile)
          return *Profile;
        if (UseCtxProfile.getNumOccurrences())
          return UseCtxProfile;
        return std::nullopt;
      }()) {}

PGOContextualProfile CtxProfAnalysis::run(Module &M,
                                          ModuleAnalysisManager &MAM) {
  if (!Profile)
    return {};
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(*Profile);
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

  DenseSet<GlobalValue::GUID> ProfileRootsInModule;
  for (const auto &F : M)
    if (!F.isDeclaration())
      if (auto GUID = AssignGUIDPass::getGUID(F);
          MaybeCtx->find(GUID) != MaybeCtx->end())
        ProfileRootsInModule.insert(GUID);

  // Trim first the roots that aren't in this module.
  for (auto &[RootGuid, _] : llvm::make_early_inc_range(*MaybeCtx))
    if (!ProfileRootsInModule.contains(RootGuid))
      MaybeCtx->erase(RootGuid);
  // If none of the roots are in the module, we have no profile (for this
  // module)
  if (MaybeCtx->empty())
    return {};

  // OK, so we have a valid profile and it's applicable to roots in this module.
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
  Result.Profiles = std::move(*MaybeCtx);
  return Result;
}

GlobalValue::GUID
PGOContextualProfile::getDefinedFunctionGUID(const Function &F) const {
  if (auto It = FuncInfo.find(AssignGUIDPass::getGUID(F)); It != FuncInfo.end())
    return It->first;
  return 0;
}

CtxProfAnalysisPrinterPass::CtxProfAnalysisPrinterPass(raw_ostream &OS)
    : OS(OS), Mode(PrintLevel) {}

PreservedAnalyses CtxProfAnalysisPrinterPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  CtxProfAnalysis::Result &C = MAM.getResult<CtxProfAnalysis>(M);
  if (!C) {
    OS << "No contextual profile was provided.\n";
    return PreservedAnalyses::all();
  }

  if (Mode == PrintMode::Everything) {
    OS << "Function Info:\n";
    for (const auto &[Guid, FuncInfo] : C.FuncInfo)
      OS << Guid << " : " << FuncInfo.Name
         << ". MaxCounterID: " << FuncInfo.NextCounterIndex
         << ". MaxCallsiteID: " << FuncInfo.NextCallsiteIndex << "\n";
  }

  const auto JSONed = ::llvm::json::toJSON(C.profiles());

  if (Mode == PrintMode::Everything)
    OS << "\nCurrent Profile:\n";
  OS << formatv("{0:2}", JSONed);
  if (Mode == PrintMode::JSON)
    return PreservedAnalyses::all();

  OS << "\n";
  OS << "\nFlat Profile:\n";
  auto Flat = C.flatten();
  for (const auto &[Guid, Counters] : Flat) {
    OS << Guid << " : ";
    for (auto V : Counters)
      OS << V << " ";
    OS << "\n";
  }
  return PreservedAnalyses::all();
}

InstrProfCallsite *CtxProfAnalysis::getCallsiteInstrumentation(CallBase &CB) {
  if (!InstrProfCallsite::canInstrumentCallsite(CB))
    return nullptr;
  for (auto *Prev = CB.getPrevNode(); Prev; Prev = Prev->getPrevNode()) {
    if (auto *IPC = dyn_cast<InstrProfCallsite>(Prev))
      return IPC;
    assert(!isa<CallBase>(Prev) &&
           "didn't expect to find another call, that's not the callsite "
           "instrumentation, before an instrumentable callsite");
  }
  return nullptr;
}

InstrProfIncrementInst *CtxProfAnalysis::getBBInstrumentation(BasicBlock &BB) {
  for (auto &I : BB)
    if (auto *Incr = dyn_cast<InstrProfIncrementInst>(&I))
      if (!isa<InstrProfIncrementInstStep>(&I))
        return Incr;
  return nullptr;
}

InstrProfIncrementInstStep *
CtxProfAnalysis::getSelectInstrumentation(SelectInst &SI) {
  Instruction *Prev = &SI;
  while ((Prev = Prev->getPrevNode()))
    if (auto *Step = dyn_cast<InstrProfIncrementInstStep>(Prev))
      return Step;
  return nullptr;
}

template <class ProfilesTy, class ProfTy>
static void preorderVisit(ProfilesTy &Profiles,
                          function_ref<void(ProfTy &)> Visitor,
                          GlobalValue::GUID Match = 0) {
  std::function<void(ProfTy &)> Traverser = [&](auto &Ctx) {
    if (!Match || Ctx.guid() == Match)
      Visitor(Ctx);
    for (auto &[_, SubCtxSet] : Ctx.callsites())
      for (auto &[__, Subctx] : SubCtxSet)
        Traverser(Subctx);
  };
  for (auto &[_, P] : Profiles)
    Traverser(P);
}

void PGOContextualProfile::update(Visitor V, const Function *F) {
  GlobalValue::GUID G = F ? getDefinedFunctionGUID(*F) : 0U;
  preorderVisit<PGOCtxProfContext::CallTargetMapTy, PGOCtxProfContext>(
      *Profiles, V, G);
}

void PGOContextualProfile::visit(ConstVisitor V, const Function *F) const {
  GlobalValue::GUID G = F ? getDefinedFunctionGUID(*F) : 0U;
  preorderVisit<const PGOCtxProfContext::CallTargetMapTy,
                const PGOCtxProfContext>(*Profiles, V, G);
}

const CtxProfFlatProfile PGOContextualProfile::flatten() const {
  assert(Profiles.has_value());
  CtxProfFlatProfile Flat;
  preorderVisit<const PGOCtxProfContext::CallTargetMapTy,
                const PGOCtxProfContext>(
      *Profiles, [&](const PGOCtxProfContext &Ctx) {
        auto [It, Ins] = Flat.insert({Ctx.guid(), {}});
        if (Ins) {
          llvm::append_range(It->second, Ctx.counters());
          return;
        }
        assert(It->second.size() == Ctx.counters().size() &&
               "All contexts corresponding to a function should have the exact "
               "same number of counters.");
        for (size_t I = 0, E = It->second.size(); I < E; ++I)
          It->second[I] += Ctx.counters()[I];
      });
  return Flat;
}

void CtxProfAnalysis::collectIndirectCallPromotionList(
    CallBase &IC, Result &Profile,
    SetVector<std::pair<CallBase *, Function *>> &Candidates) {
  const auto *Instr = CtxProfAnalysis::getCallsiteInstrumentation(IC);
  if (!Instr)
    return;
  Module &M = *IC.getParent()->getModule();
  const uint32_t CallID = Instr->getIndex()->getZExtValue();
  Profile.visit(
      [&](const PGOCtxProfContext &Ctx) {
        const auto &Targets = Ctx.callsites().find(CallID);
        if (Targets == Ctx.callsites().end())
          return;
        for (const auto &[Guid, _] : Targets->second)
          if (auto Name = Profile.getFunctionName(Guid); !Name.empty())
            if (auto *Target = M.getFunction(Name))
              if (Target->hasFnAttribute(Attribute::AlwaysInline))
                Candidates.insert({&IC, Target});
      },
      IC.getCaller());
}
