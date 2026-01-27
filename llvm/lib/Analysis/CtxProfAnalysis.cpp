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
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <deque>
#include <memory>

#define DEBUG_TYPE "ctx_prof"

using namespace llvm;

namespace llvm {

cl::opt<std::string>
    UseCtxProfile("use-ctx-profile", cl::init(""), cl::Hidden,
                  cl::desc("Use the specified contextual profile file"));

static cl::opt<CtxProfAnalysisPrinterPass::PrintMode> PrintLevel(
    "ctx-profile-printer-level",
    cl::init(CtxProfAnalysisPrinterPass::PrintMode::YAML), cl::Hidden,
    cl::values(clEnumValN(CtxProfAnalysisPrinterPass::PrintMode::Everything,
                          "everything", "print everything - most verbose"),
               clEnumValN(CtxProfAnalysisPrinterPass::PrintMode::YAML, "yaml",
                          "just the yaml representation of the profile")),
    cl::desc("Verbosity level of the contextual profile printer pass."));

static cl::opt<bool> ForceIsInSpecializedModule(
    "ctx-profile-force-is-specialized", cl::init(false),
    cl::desc("Treat the given module as-if it were containing the "
             "post-thinlink module containing the root"));

const char *AssignGUIDPass::GUIDMetadataName = "guid";

class ProfileAnnotatorImpl final {
  friend class ProfileAnnotator;
  class BBInfo;
  struct EdgeInfo {
    BBInfo *const Src;
    BBInfo *const Dest;
    std::optional<uint64_t> Count;

    explicit EdgeInfo(BBInfo &Src, BBInfo &Dest) : Src(&Src), Dest(&Dest) {}
  };

  class BBInfo {
    std::optional<uint64_t> Count;
    // OutEdges is dimensioned to match the number of terminator operands.
    // Entries in the vector match the index in the terminator operand list. In
    // some cases - see `shouldExcludeEdge` and its implementation - an entry
    // will be nullptr.
    // InEdges doesn't have the above constraint.
    SmallVector<EdgeInfo *> OutEdges;
    SmallVector<EdgeInfo *> InEdges;
    size_t UnknownCountOutEdges = 0;
    size_t UnknownCountInEdges = 0;

    // Pass AssumeAllKnown when we try to propagate counts from edges to BBs -
    // because all the edge counters must be known.
    // Return std::nullopt if there were no edges to sum. The user can decide
    // how to interpret that.
    std::optional<uint64_t> getEdgeSum(const SmallVector<EdgeInfo *> &Edges,
                                       bool AssumeAllKnown) const {
      std::optional<uint64_t> Sum;
      for (const auto *E : Edges) {
        // `Edges` may be `OutEdges`, case in which `E` could be nullptr.
        if (E) {
          if (!Sum.has_value())
            Sum = 0;
          *Sum += (AssumeAllKnown ? *E->Count : E->Count.value_or(0U));
        }
      }
      return Sum;
    }

    bool computeCountFrom(const SmallVector<EdgeInfo *> &Edges) {
      assert(!Count.has_value());
      Count = getEdgeSum(Edges, true);
      return Count.has_value();
    }

    void setSingleUnknownEdgeCount(SmallVector<EdgeInfo *> &Edges) {
      uint64_t KnownSum = getEdgeSum(Edges, false).value_or(0U);
      uint64_t EdgeVal = *Count > KnownSum ? *Count - KnownSum : 0U;
      EdgeInfo *E = nullptr;
      for (auto *I : Edges)
        if (I && !I->Count.has_value()) {
          E = I;
#ifdef NDEBUG
          break;
#else
          assert((!E || E == I) &&
                 "Expected exactly one edge to have an unknown count, "
                 "found a second one");
          continue;
#endif
        }
      assert(E && "Expected exactly one edge to have an unknown count");
      assert(!E->Count.has_value());
      E->Count = EdgeVal;
      assert(E->Src->UnknownCountOutEdges > 0);
      assert(E->Dest->UnknownCountInEdges > 0);
      --E->Src->UnknownCountOutEdges;
      --E->Dest->UnknownCountInEdges;
    }

  public:
    BBInfo(size_t NumInEdges, size_t NumOutEdges, std::optional<uint64_t> Count)
        : Count(Count) {
      // For in edges, we just want to pre-allocate enough space, since we know
      // it at this stage. For out edges, we will insert edges at the indices
      // corresponding to positions in this BB's terminator instruction, so we
      // construct a default (nullptr values)-initialized vector. A nullptr edge
      // corresponds to those that are excluded (see shouldExcludeEdge).
      InEdges.reserve(NumInEdges);
      OutEdges.resize(NumOutEdges);
    }

    bool tryTakeCountFromKnownOutEdges(const BasicBlock &BB) {
      if (!UnknownCountOutEdges) {
        return computeCountFrom(OutEdges);
      }
      return false;
    }

    bool tryTakeCountFromKnownInEdges(const BasicBlock &BB) {
      if (!UnknownCountInEdges) {
        return computeCountFrom(InEdges);
      }
      return false;
    }

    void addInEdge(EdgeInfo &Info) {
      InEdges.push_back(&Info);
      ++UnknownCountInEdges;
    }

    // For the out edges, we care about the position we place them in, which is
    // the position in terminator instruction's list (at construction). Later,
    // we build branch_weights metadata with edge frequency values matching
    // these positions.
    void addOutEdge(size_t Index, EdgeInfo &Info) {
      OutEdges[Index] = &Info;
      ++UnknownCountOutEdges;
    }

    bool hasCount() const { return Count.has_value(); }

    uint64_t getCount() const { return *Count; }

    bool trySetSingleUnknownInEdgeCount() {
      if (UnknownCountInEdges == 1) {
        setSingleUnknownEdgeCount(InEdges);
        return true;
      }
      return false;
    }

    bool trySetSingleUnknownOutEdgeCount() {
      if (UnknownCountOutEdges == 1) {
        setSingleUnknownEdgeCount(OutEdges);
        return true;
      }
      return false;
    }
    size_t getNumOutEdges() const { return OutEdges.size(); }

    uint64_t getEdgeCount(size_t Index) const {
      if (auto *E = OutEdges[Index])
        return *E->Count;
      return 0U;
    }
  };

  const Function &F;
  ArrayRef<uint64_t> Counters;
  // To be accessed through getBBInfo() after construction.
  std::map<const BasicBlock *, BBInfo> BBInfos;
  std::vector<EdgeInfo> EdgeInfos;

  // The only criteria for exclusion is faux suspend -> exit edges in presplit
  // coroutines. The API serves for readability, currently.
  bool shouldExcludeEdge(const BasicBlock &Src, const BasicBlock &Dest) const {
    return llvm::isPresplitCoroSuspendExitEdge(Src, Dest);
  }

  BBInfo &getBBInfo(const BasicBlock &BB) { return BBInfos.find(&BB)->second; }

  const BBInfo &getBBInfo(const BasicBlock &BB) const {
    return BBInfos.find(&BB)->second;
  }

  // validation function after we propagate the counters: all BBs and edges'
  // counters must have a value.
  bool allCountersAreAssigned() const {
    for (const auto &BBInfo : BBInfos)
      if (!BBInfo.second.hasCount())
        return false;
    for (const auto &EdgeInfo : EdgeInfos)
      if (!EdgeInfo.Count.has_value())
        return false;
    return true;
  }

  /// Check that all paths from the entry basic block that use edges with
  /// non-zero counts arrive at a basic block with no successors (i.e. "exit")
  bool allTakenPathsExit() const {
    std::deque<const BasicBlock *> Worklist;
    DenseSet<const BasicBlock *> Visited;
    Worklist.push_back(&F.getEntryBlock());
    bool HitExit = false;
    while (!Worklist.empty()) {
      const auto *BB = Worklist.front();
      Worklist.pop_front();
      if (!Visited.insert(BB).second)
        continue;
      if (succ_size(BB) == 0) {
        if (isa<UnreachableInst>(BB->getTerminator()))
          return false;
        HitExit = true;
        continue;
      }
      if (succ_size(BB) == 1) {
        Worklist.push_back(BB->getUniqueSuccessor());
        continue;
      }
      const auto &BBInfo = getBBInfo(*BB);
      bool HasAWayOut = false;
      for (auto I = 0U; I < BB->getTerminator()->getNumSuccessors(); ++I) {
        const auto *Succ = BB->getTerminator()->getSuccessor(I);
        if (!shouldExcludeEdge(*BB, *Succ)) {
          if (BBInfo.getEdgeCount(I) > 0) {
            HasAWayOut = true;
            Worklist.push_back(Succ);
          }
        }
      }
      if (!HasAWayOut)
        return false;
    }
    return HitExit;
  }

  bool allNonColdSelectsHaveProfile() const {
    for (const auto &BB : F) {
      if (getBBInfo(BB).getCount() > 0) {
        for (const auto &I : BB) {
          if (const auto *SI = dyn_cast<SelectInst>(&I)) {
            if (const auto *Inst = CtxProfAnalysis::getSelectInstrumentation(
                    *const_cast<SelectInst *>(SI))) {
              auto Index = Inst->getIndex()->getZExtValue();
              assert(Index < Counters.size());
              if (Counters[Index] == 0)
                return false;
            }
          }
        }
      }
    }
    return true;
  }

  // This is an adaptation of PGOUseFunc::populateCounters.
  // FIXME(mtrofin): look into factoring the code to share one implementation.
  void propagateCounterValues() {
    bool KeepGoing = true;
    while (KeepGoing) {
      KeepGoing = false;
      for (const auto &BB : F) {
        auto &Info = getBBInfo(BB);
        if (!Info.hasCount())
          KeepGoing |= Info.tryTakeCountFromKnownOutEdges(BB) ||
                       Info.tryTakeCountFromKnownInEdges(BB);
        if (Info.hasCount()) {
          KeepGoing |= Info.trySetSingleUnknownOutEdgeCount();
          KeepGoing |= Info.trySetSingleUnknownInEdgeCount();
        }
      }
    }
    assert(allCountersAreAssigned() &&
           "[ctx-prof] Expected all counters have been assigned.");
    assert(allTakenPathsExit() &&
           "[ctx-prof] Encountered a BB with more than one successor, where "
           "all outgoing edges have a 0 count. This occurs in non-exiting "
           "functions (message pumps, usually) which are not supported in the "
           "contextual profiling case");
    assert(allNonColdSelectsHaveProfile() &&
           "[ctx-prof] All non-cold select instructions were expected to have "
           "a profile.");
  }

public:
  ProfileAnnotatorImpl(const Function &F, ArrayRef<uint64_t> Counters)
      : F(F), Counters(Counters) {
    assert(!F.isDeclaration());
    assert(!Counters.empty());
    size_t NrEdges = 0;
    for (const auto &BB : F) {
      std::optional<uint64_t> Count;
      if (auto *Ins = CtxProfAnalysis::getBBInstrumentation(
              const_cast<BasicBlock &>(BB))) {
        auto Index = Ins->getIndex()->getZExtValue();
        assert(Index < Counters.size() &&
               "The index must be inside the counters vector by construction - "
               "tripping this assertion indicates a bug in how the contextual "
               "profile is managed by IPO transforms");
        (void)Index;
        Count = Counters[Ins->getIndex()->getZExtValue()];
      } else if (isa<UnreachableInst>(BB.getTerminator())) {
        // The program presumably didn't crash.
        Count = 0;
      }
      auto [It, Ins] =
          BBInfos.insert({&BB, {pred_size(&BB), succ_size(&BB), Count}});
      (void)Ins;
      assert(Ins && "We iterate through the function's BBs, no reason to "
                    "insert one more than once");
      NrEdges += llvm::count_if(successors(&BB), [&](const auto *Succ) {
        return !shouldExcludeEdge(BB, *Succ);
      });
    }
    // Pre-allocate the vector, we want references to its contents to be stable.
    EdgeInfos.reserve(NrEdges);
    for (const auto &BB : F) {
      auto &Info = getBBInfo(BB);
      for (auto I = 0U; I < BB.getTerminator()->getNumSuccessors(); ++I) {
        const auto *Succ = BB.getTerminator()->getSuccessor(I);
        if (!shouldExcludeEdge(BB, *Succ)) {
          auto &EI = EdgeInfos.emplace_back(getBBInfo(BB), getBBInfo(*Succ));
          Info.addOutEdge(I, EI);
          getBBInfo(*Succ).addInEdge(EI);
        }
      }
    }
    assert(EdgeInfos.capacity() == NrEdges &&
           "The capacity of EdgeInfos should have stayed unchanged it was "
           "populated, because we need pointers to its contents to be stable");
    propagateCounterValues();
  }

  uint64_t getBBCount(const BasicBlock &BB) { return getBBInfo(BB).getCount(); }
};

} // namespace llvm

ProfileAnnotator::ProfileAnnotator(const Function &F,
                                   ArrayRef<uint64_t> RawCounters)
    : PImpl(std::make_unique<ProfileAnnotatorImpl>(F, RawCounters)) {}

ProfileAnnotator::~ProfileAnnotator() = default;

uint64_t ProfileAnnotator::getBBCount(const BasicBlock &BB) const {
  return PImpl->getBBCount(BB);
}

bool ProfileAnnotator::getSelectInstrProfile(SelectInst &SI,
                                             uint64_t &TrueCount,
                                             uint64_t &FalseCount) const {
  const auto &BBInfo = PImpl->getBBInfo(*SI.getParent());
  TrueCount = FalseCount = 0;
  if (BBInfo.getCount() == 0)
    return false;

  auto *Step = CtxProfAnalysis::getSelectInstrumentation(SI);
  if (!Step)
    return false;
  auto Index = Step->getIndex()->getZExtValue();
  assert(Index < PImpl->Counters.size() &&
         "The index of the step instruction must be inside the "
         "counters vector by "
         "construction - tripping this assertion indicates a bug in "
         "how the contextual profile is managed by IPO transforms");
  auto TotalCount = BBInfo.getCount();
  TrueCount = PImpl->Counters[Index];
  FalseCount = (TotalCount > TrueCount ? TotalCount - TrueCount : 0U);
  return true;
}

bool ProfileAnnotator::getOutgoingBranchWeights(
    BasicBlock &BB, SmallVectorImpl<uint64_t> &Profile,
    uint64_t &MaxCount) const {
  Profile.clear();

  if (succ_size(&BB) < 2)
    return false;

  auto *Term = BB.getTerminator();
  Profile.resize(Term->getNumSuccessors());

  const auto &BBInfo = PImpl->getBBInfo(BB);
  MaxCount = 0;
  for (unsigned SuccIdx = 0, Size = BBInfo.getNumOutEdges(); SuccIdx < Size;
       ++SuccIdx) {
    uint64_t EdgeCount = BBInfo.getEdgeCount(SuccIdx);
    if (EdgeCount > MaxCount)
      MaxCount = EdgeCount;
    Profile[SuccIdx] = EdgeCount;
  }
  return MaxCount > 0;
}

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
    return F.getGUID();
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
  auto MaybeProfiles = Reader.loadProfiles();
  if (!MaybeProfiles) {
    M.getContext().emitError("contextual profile file is invalid: " +
                             toString(MaybeProfiles.takeError()));
    return {};
  }

  // FIXME: We should drive this from ThinLTO, but for the time being, use the
  // module name as indicator.
  // We want to *only* keep the contextual profiles in modules that capture
  // context trees. That allows us to compute specific PSIs, for example.
  auto DetermineRootsInModule = [&M]() -> const DenseSet<GlobalValue::GUID> {
    DenseSet<GlobalValue::GUID> ProfileRootsInModule;
    auto ModName = M.getName();
    auto Filename = sys::path::filename(ModName);
    // Drop the file extension.
    Filename = Filename.substr(0, Filename.find_last_of('.'));
    // See if it parses
    APInt Guid;
    // getAsInteger returns true if there are more chars to read other than the
    // integer. So the "false" test is what we want.
    if (!Filename.getAsInteger(0, Guid))
      ProfileRootsInModule.insert(Guid.getZExtValue());
    return ProfileRootsInModule;
  };
  const auto ProfileRootsInModule = DetermineRootsInModule();
  PGOContextualProfile Result;

  // the logic from here on allows for modules that contain - by design - more
  // than one root. We currently don't support that, because the determination
  // happens based on the module name matching the root guid, but the logic can
  // avoid assuming that.
  if (!ProfileRootsInModule.empty()) {
    Result.IsInSpecializedModule = true;
    // Trim first the roots that aren't in this module.
    for (auto &[RootGuid, _] :
         llvm::make_early_inc_range(MaybeProfiles->Contexts))
      if (!ProfileRootsInModule.contains(RootGuid))
        MaybeProfiles->Contexts.erase(RootGuid);
    // we can also drop the flat profiles
    MaybeProfiles->FlatProfiles.clear();
  }

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
  Result.Profiles = std::move(*MaybeProfiles);
  Result.initIndex();
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
  if (C.contexts().empty()) {
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

  if (Mode == PrintMode::Everything)
    OS << "\nCurrent Profile:\n";
  convertCtxProfToYaml(OS, C.profiles());
  OS << "\n";
  if (Mode == PrintMode::YAML)
    return PreservedAnalyses::all();

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

template <class ProfTy>
static void preorderVisitOneRoot(ProfTy &Profile,
                                 function_ref<void(ProfTy &)> Visitor) {
  std::function<void(ProfTy &)> Traverser = [&](auto &Ctx) {
    Visitor(Ctx);
    for (auto &[_, SubCtxSet] : Ctx.callsites())
      for (auto &[__, Subctx] : SubCtxSet)
        Traverser(Subctx);
  };
  Traverser(Profile);
}

template <class ProfilesTy, class ProfTy>
static void preorderVisit(ProfilesTy &Profiles,
                          function_ref<void(ProfTy &)> Visitor) {
  for (auto &[_, P] : Profiles)
    preorderVisitOneRoot<ProfTy>(P, Visitor);
}

void PGOContextualProfile::initIndex() {
  // Initialize the head of the index list for each function. We don't need it
  // after this point.
  DenseMap<GlobalValue::GUID, PGOCtxProfContext *> InsertionPoints;
  for (auto &[Guid, FI] : FuncInfo)
    InsertionPoints[Guid] = &FI.Index;
  preorderVisit<PGOCtxProfContext::CallTargetMapTy, PGOCtxProfContext>(
      Profiles.Contexts, [&](PGOCtxProfContext &Ctx) {
        auto InsertIt = InsertionPoints.find(Ctx.guid());
        if (InsertIt == InsertionPoints.end())
          return;
        // Insert at the end of the list. Since we traverse in preorder, it
        // means that when we iterate the list from the beginning, we'd
        // encounter the contexts in the order we would have, should we have
        // performed a full preorder traversal.
        InsertIt->second->Next = &Ctx;
        Ctx.Previous = InsertIt->second;
        InsertIt->second = &Ctx;
      });
}

bool PGOContextualProfile::isInSpecializedModule() const {
  return ForceIsInSpecializedModule.getNumOccurrences() > 0
             ? ForceIsInSpecializedModule
             : IsInSpecializedModule;
}

void PGOContextualProfile::update(Visitor V, const Function &F) {
  assert(isFunctionKnown(F));
  GlobalValue::GUID G = getDefinedFunctionGUID(F);
  for (auto *Node = FuncInfo.find(G)->second.Index.Next; Node;
       Node = Node->Next)
    V(*reinterpret_cast<PGOCtxProfContext *>(Node));
}

void PGOContextualProfile::visit(ConstVisitor V, const Function *F) const {
  if (!F)
    return preorderVisit<const PGOCtxProfContext::CallTargetMapTy,
                         const PGOCtxProfContext>(Profiles.Contexts, V);
  assert(isFunctionKnown(*F));
  GlobalValue::GUID G = getDefinedFunctionGUID(*F);
  for (const auto *Node = FuncInfo.find(G)->second.Index.Next; Node;
       Node = Node->Next)
    V(*reinterpret_cast<const PGOCtxProfContext *>(Node));
}

const CtxProfFlatProfile PGOContextualProfile::flatten() const {
  CtxProfFlatProfile Flat;
  auto Accummulate = [](SmallVectorImpl<uint64_t> &Into,
                        const SmallVectorImpl<uint64_t> &From,
                        uint64_t SamplingRate) {
    if (Into.empty())
      Into.resize(From.size());
    assert(Into.size() == From.size() &&
           "All contexts corresponding to a function should have the exact "
           "same number of counters.");
    for (size_t I = 0, E = Into.size(); I < E; ++I)
      Into[I] += From[I] * SamplingRate;
  };

  for (const auto &[_, CtxRoot] : Profiles.Contexts) {
    const uint64_t SamplingFactor = CtxRoot.getTotalRootEntryCount();
    preorderVisitOneRoot<const PGOCtxProfContext>(
        CtxRoot, [&](const PGOCtxProfContext &Ctx) {
          Accummulate(Flat[Ctx.guid()], Ctx.counters(), SamplingFactor);
        });

    for (const auto &[G, Unh] : CtxRoot.getUnhandled())
      Accummulate(Flat[G], Unh, SamplingFactor);
  }
  // We don't sample "Flat" currently, so sampling rate is 1.
  for (const auto &[G, FC] : Profiles.FlatProfiles)
    Accummulate(Flat[G], FC, /*SamplingRate=*/1);
  return Flat;
}

const CtxProfFlatIndirectCallProfile
PGOContextualProfile::flattenVirtCalls() const {
  CtxProfFlatIndirectCallProfile Ret;
  for (const auto &[_, CtxRoot] : Profiles.Contexts) {
    const uint64_t TotalRootEntryCount = CtxRoot.getTotalRootEntryCount();
    preorderVisitOneRoot<const PGOCtxProfContext>(
        CtxRoot, [&](const PGOCtxProfContext &Ctx) {
          auto &Targets = Ret[Ctx.guid()];
          for (const auto &[ID, SubctxSet] : Ctx.callsites())
            for (const auto &Subctx : SubctxSet)
              Targets[ID][Subctx.first] +=
                  Subctx.second.getEntrycount() * TotalRootEntryCount;
        });
  }
  return Ret;
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
