//===-- InstrProfiling.cpp - Frontend instrumentation based profiling -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers instrprof_* intrinsics emitted by a frontend for profiling.
// It also builds the data structures and initialization code needed for
// updating execution counts and emitting the profile at runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfCorrelator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "instrprof"

namespace llvm {
cl::opt<bool>
    DebugInfoCorrelate("debug-info-correlate",
                       cl::desc("Use debug info to correlate profiles."),
                       cl::init(false));
} // namespace llvm

namespace {

cl::opt<bool> DoHashBasedCounterSplit(
    "hash-based-counter-split",
    cl::desc("Rename counter variable of a comdat function based on cfg hash"),
    cl::init(true));

cl::opt<bool>
    RuntimeCounterRelocation("runtime-counter-relocation",
                             cl::desc("Enable relocating counters at runtime."),
                             cl::init(false));

cl::opt<bool> ValueProfileStaticAlloc(
    "vp-static-alloc",
    cl::desc("Do static counter allocation for value profiler"),
    cl::init(true));

cl::opt<double> NumCountersPerValueSite(
    "vp-counters-per-site",
    cl::desc("The average number of profile counters allocated "
             "per value profiling site."),
    // This is set to a very small value because in real programs, only
    // a very small percentage of value sites have non-zero targets, e.g, 1/30.
    // For those sites with non-zero profile, the average number of targets
    // is usually smaller than 2.
    cl::init(1.0));

cl::opt<bool> AtomicCounterUpdateAll(
    "instrprof-atomic-counter-update-all",
    cl::desc("Make all profile counter updates atomic (for testing only)"),
    cl::init(false));

cl::opt<bool> AtomicCounterUpdatePromoted(
    "atomic-counter-update-promoted",
    cl::desc("Do counter update using atomic fetch add "
             " for promoted counters only"),
    cl::init(false));

cl::opt<bool> AtomicFirstCounter(
    "atomic-first-counter",
    cl::desc("Use atomic fetch add for first counter in a function (usually "
             "the entry counter)"),
    cl::init(false));

// If the option is not specified, the default behavior about whether
// counter promotion is done depends on how instrumentaiton lowering
// pipeline is setup, i.e., the default value of true of this option
// does not mean the promotion will be done by default. Explicitly
// setting this option can override the default behavior.
cl::opt<bool> DoCounterPromotion("do-counter-promotion",
                                 cl::desc("Do counter register promotion"),
                                 cl::init(false));
cl::opt<unsigned> MaxNumOfPromotionsPerLoop(
    cl::ZeroOrMore, "max-counter-promotions-per-loop", cl::init(20),
    cl::desc("Max number counter promotions per loop to avoid"
             " increasing register pressure too much"));

// A debug option
cl::opt<int>
    MaxNumOfPromotions(cl::ZeroOrMore, "max-counter-promotions", cl::init(-1),
                       cl::desc("Max number of allowed counter promotions"));

cl::opt<unsigned> SpeculativeCounterPromotionMaxExiting(
    cl::ZeroOrMore, "speculative-counter-promotion-max-exiting", cl::init(3),
    cl::desc("The max number of exiting blocks of a loop to allow "
             " speculative counter promotion"));

cl::opt<bool> SpeculativeCounterPromotionToLoop(
    cl::ZeroOrMore, "speculative-counter-promotion-to-loop", cl::init(false),
    cl::desc("When the option is false, if the target block is in a loop, "
             "the promotion will be disallowed unless the promoted counter "
             " update can be further/iteratively promoted into an acyclic "
             " region."));

cl::opt<bool> IterativeCounterPromotion(
    cl::ZeroOrMore, "iterative-counter-promotion", cl::init(true),
    cl::desc("Allow counter promotion across the whole loop nest."));

cl::opt<bool> SkipRetExitBlock(
    cl::ZeroOrMore, "skip-ret-exit-block", cl::init(true),
    cl::desc("Suppress counter promotion if exit blocks contain ret."));

class InstrProfilingLegacyPass : public ModulePass {
  InstrProfiling InstrProf;

public:
  static char ID;

  InstrProfilingLegacyPass() : ModulePass(ID) {}
  InstrProfilingLegacyPass(const InstrProfOptions &Options, bool IsCS = false)
      : ModulePass(ID), InstrProf(Options, IsCS) {
    initializeInstrProfilingLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Frontend instrumentation-based coverage lowering";
  }

  bool runOnModule(Module &M) override {
    auto GetTLI = [this](Function &F) -> TargetLibraryInfo & {
      return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };
    return InstrProf.run(M, GetTLI);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};

///
/// A helper class to promote one counter RMW operation in the loop
/// into register update.
///
/// RWM update for the counter will be sinked out of the loop after
/// the transformation.
///
class PGOCounterPromoterHelper : public LoadAndStorePromoter {
public:
  PGOCounterPromoterHelper(
      Instruction *L, Instruction *S, SSAUpdater &SSA, Value *Init,
      BasicBlock *PH, ArrayRef<BasicBlock *> ExitBlocks,
      ArrayRef<Instruction *> InsertPts,
      DenseMap<Loop *, SmallVector<LoadStorePair, 8>> &LoopToCands,
      LoopInfo &LI)
      : LoadAndStorePromoter({L, S}, SSA), Store(S), ExitBlocks(ExitBlocks),
        InsertPts(InsertPts), LoopToCandidates(LoopToCands), LI(LI) {
    assert(isa<LoadInst>(L));
    assert(isa<StoreInst>(S));
    SSA.AddAvailableValue(PH, Init);
  }

  void doExtraRewritesBeforeFinalDeletion() override {
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
      BasicBlock *ExitBlock = ExitBlocks[i];
      Instruction *InsertPos = InsertPts[i];
      // Get LiveIn value into the ExitBlock. If there are multiple
      // predecessors, the value is defined by a PHI node in this
      // block.
      Value *LiveInValue = SSA.GetValueInMiddleOfBlock(ExitBlock);
      Value *Addr = cast<StoreInst>(Store)->getPointerOperand();
      Type *Ty = LiveInValue->getType();
      IRBuilder<> Builder(InsertPos);
      if (auto *AddrInst = dyn_cast_or_null<IntToPtrInst>(Addr)) {
        // If isRuntimeCounterRelocationEnabled() is true then the address of
        // the store instruction is computed with two instructions in
        // InstrProfiling::getCounterAddress(). We need to copy those
        // instructions to this block to compute Addr correctly.
        // %BiasAdd = add i64 ptrtoint <__profc_>, <__llvm_profile_counter_bias>
        // %Addr = inttoptr i64 %BiasAdd to i64*
        auto *OrigBiasInst = dyn_cast<BinaryOperator>(AddrInst->getOperand(0));
        assert(OrigBiasInst->getOpcode() == Instruction::BinaryOps::Add);
        Value *BiasInst = Builder.Insert(OrigBiasInst->clone());
        Addr = Builder.CreateIntToPtr(BiasInst, Ty->getPointerTo());
      }
      if (AtomicCounterUpdatePromoted)
        // automic update currently can only be promoted across the current
        // loop, not the whole loop nest.
        Builder.CreateAtomicRMW(AtomicRMWInst::Add, Addr, LiveInValue,
                                MaybeAlign(),
                                AtomicOrdering::SequentiallyConsistent);
      else {
        LoadInst *OldVal = Builder.CreateLoad(Ty, Addr, "pgocount.promoted");
        auto *NewVal = Builder.CreateAdd(OldVal, LiveInValue);
        auto *NewStore = Builder.CreateStore(NewVal, Addr);

        // Now update the parent loop's candidate list:
        if (IterativeCounterPromotion) {
          auto *TargetLoop = LI.getLoopFor(ExitBlock);
          if (TargetLoop)
            LoopToCandidates[TargetLoop].emplace_back(OldVal, NewStore);
        }
      }
    }
  }

private:
  Instruction *Store;
  ArrayRef<BasicBlock *> ExitBlocks;
  ArrayRef<Instruction *> InsertPts;
  DenseMap<Loop *, SmallVector<LoadStorePair, 8>> &LoopToCandidates;
  LoopInfo &LI;
};

/// A helper class to do register promotion for all profile counter
/// updates in a loop.
///
class PGOCounterPromoter {
public:
  PGOCounterPromoter(
      DenseMap<Loop *, SmallVector<LoadStorePair, 8>> &LoopToCands,
      Loop &CurLoop, LoopInfo &LI, BlockFrequencyInfo *BFI)
      : LoopToCandidates(LoopToCands), L(CurLoop), LI(LI), BFI(BFI) {

    // Skip collection of ExitBlocks and InsertPts for loops that will not be
    // able to have counters promoted.
    SmallVector<BasicBlock *, 8> LoopExitBlocks;
    SmallPtrSet<BasicBlock *, 8> BlockSet;

    L.getExitBlocks(LoopExitBlocks);
    if (!isPromotionPossible(&L, LoopExitBlocks))
      return;

    for (BasicBlock *ExitBlock : LoopExitBlocks) {
      if (BlockSet.insert(ExitBlock).second) {
        ExitBlocks.push_back(ExitBlock);
        InsertPts.push_back(&*ExitBlock->getFirstInsertionPt());
      }
    }
  }

  bool run(int64_t *NumPromoted) {
    // Skip 'infinite' loops:
    if (ExitBlocks.size() == 0)
      return false;

    // Skip if any of the ExitBlocks contains a ret instruction.
    // This is to prevent dumping of incomplete profile -- if the
    // the loop is a long running loop and dump is called in the middle
    // of the loop, the result profile is incomplete.
    // FIXME: add other heuristics to detect long running loops.
    if (SkipRetExitBlock) {
      for (auto BB : ExitBlocks)
        if (isa<ReturnInst>(BB->getTerminator()))
          return false;
    }

    unsigned MaxProm = getMaxNumOfPromotionsInLoop(&L);
    if (MaxProm == 0)
      return false;

    unsigned Promoted = 0;
    for (auto &Cand : LoopToCandidates[&L]) {

      SmallVector<PHINode *, 4> NewPHIs;
      SSAUpdater SSA(&NewPHIs);
      Value *InitVal = ConstantInt::get(Cand.first->getType(), 0);

      // If BFI is set, we will use it to guide the promotions.
      if (BFI) {
        auto *BB = Cand.first->getParent();
        auto InstrCount = BFI->getBlockProfileCount(BB);
        if (!InstrCount)
          continue;
        auto PreheaderCount = BFI->getBlockProfileCount(L.getLoopPreheader());
        // If the average loop trip count is not greater than 1.5, we skip
        // promotion.
        if (PreheaderCount &&
            (PreheaderCount.getValue() * 3) >= (InstrCount.getValue() * 2))
          continue;
      }

      PGOCounterPromoterHelper Promoter(Cand.first, Cand.second, SSA, InitVal,
                                        L.getLoopPreheader(), ExitBlocks,
                                        InsertPts, LoopToCandidates, LI);
      Promoter.run(SmallVector<Instruction *, 2>({Cand.first, Cand.second}));
      Promoted++;
      if (Promoted >= MaxProm)
        break;

      (*NumPromoted)++;
      if (MaxNumOfPromotions != -1 && *NumPromoted >= MaxNumOfPromotions)
        break;
    }

    LLVM_DEBUG(dbgs() << Promoted << " counters promoted for loop (depth="
                      << L.getLoopDepth() << ")\n");
    return Promoted != 0;
  }

private:
  bool allowSpeculativeCounterPromotion(Loop *LP) {
    SmallVector<BasicBlock *, 8> ExitingBlocks;
    L.getExitingBlocks(ExitingBlocks);
    // Not considierered speculative.
    if (ExitingBlocks.size() == 1)
      return true;
    if (ExitingBlocks.size() > SpeculativeCounterPromotionMaxExiting)
      return false;
    return true;
  }

  // Check whether the loop satisfies the basic conditions needed to perform
  // Counter Promotions.
  bool
  isPromotionPossible(Loop *LP,
                      const SmallVectorImpl<BasicBlock *> &LoopExitBlocks) {
    // We can't insert into a catchswitch.
    if (llvm::any_of(LoopExitBlocks, [](BasicBlock *Exit) {
          return isa<CatchSwitchInst>(Exit->getTerminator());
        }))
      return false;

    if (!LP->hasDedicatedExits())
      return false;

    BasicBlock *PH = LP->getLoopPreheader();
    if (!PH)
      return false;

    return true;
  }

  // Returns the max number of Counter Promotions for LP.
  unsigned getMaxNumOfPromotionsInLoop(Loop *LP) {
    SmallVector<BasicBlock *, 8> LoopExitBlocks;
    LP->getExitBlocks(LoopExitBlocks);
    if (!isPromotionPossible(LP, LoopExitBlocks))
      return 0;

    SmallVector<BasicBlock *, 8> ExitingBlocks;
    LP->getExitingBlocks(ExitingBlocks);

    // If BFI is set, we do more aggressive promotions based on BFI.
    if (BFI)
      return (unsigned)-1;

    // Not considierered speculative.
    if (ExitingBlocks.size() == 1)
      return MaxNumOfPromotionsPerLoop;

    if (ExitingBlocks.size() > SpeculativeCounterPromotionMaxExiting)
      return 0;

    // Whether the target block is in a loop does not matter:
    if (SpeculativeCounterPromotionToLoop)
      return MaxNumOfPromotionsPerLoop;

    // Now check the target block:
    unsigned MaxProm = MaxNumOfPromotionsPerLoop;
    for (auto *TargetBlock : LoopExitBlocks) {
      auto *TargetLoop = LI.getLoopFor(TargetBlock);
      if (!TargetLoop)
        continue;
      unsigned MaxPromForTarget = getMaxNumOfPromotionsInLoop(TargetLoop);
      unsigned PendingCandsInTarget = LoopToCandidates[TargetLoop].size();
      MaxProm =
          std::min(MaxProm, std::max(MaxPromForTarget, PendingCandsInTarget) -
                                PendingCandsInTarget);
    }
    return MaxProm;
  }

  DenseMap<Loop *, SmallVector<LoadStorePair, 8>> &LoopToCandidates;
  SmallVector<BasicBlock *, 8> ExitBlocks;
  SmallVector<Instruction *, 8> InsertPts;
  Loop &L;
  LoopInfo &LI;
  BlockFrequencyInfo *BFI;
};

enum class ValueProfilingCallType {
  // Individual values are tracked. Currently used for indiret call target
  // profiling.
  Default,

  // MemOp: the memop size value profiling.
  MemOp
};

} // end anonymous namespace

PreservedAnalyses InstrProfiling::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  if (!run(M, GetTLI))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

char InstrProfilingLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(InstrProfilingLegacyPass, "instrprof",
                      "Frontend instrumentation-based coverage lowering.",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(InstrProfilingLegacyPass, "instrprof",
                    "Frontend instrumentation-based coverage lowering.", false,
                    false)

ModulePass *
llvm::createInstrProfilingLegacyPass(const InstrProfOptions &Options,
                                     bool IsCS) {
  return new InstrProfilingLegacyPass(Options, IsCS);
}

bool InstrProfiling::lowerIntrinsics(Function *F) {
  bool MadeChange = false;
  PromotionCandidates.clear();
  for (BasicBlock &BB : *F) {
    for (Instruction &Instr : llvm::make_early_inc_range(BB)) {
      if (auto *IPIS = dyn_cast<InstrProfIncrementInstStep>(&Instr)) {
        lowerIncrement(IPIS);
        MadeChange = true;
      } else if (auto *IPI = dyn_cast<InstrProfIncrementInst>(&Instr)) {
        lowerIncrement(IPI);
        MadeChange = true;
      } else if (auto *IPC = dyn_cast<InstrProfCoverInst>(&Instr)) {
        lowerCover(IPC);
        MadeChange = true;
      } else if (auto *IPVP = dyn_cast<InstrProfValueProfileInst>(&Instr)) {
        lowerValueProfileInst(IPVP);
        MadeChange = true;
      }
    }
  }

  if (!MadeChange)
    return false;

  promoteCounterLoadStores(F);
  return true;
}

bool InstrProfiling::isRuntimeCounterRelocationEnabled() const {
  // Mach-O don't support weak external references.
  if (TT.isOSBinFormatMachO())
    return false;

  if (RuntimeCounterRelocation.getNumOccurrences() > 0)
    return RuntimeCounterRelocation;

  // Fuchsia uses runtime counter relocation by default.
  return TT.isOSFuchsia();
}

bool InstrProfiling::isCounterPromotionEnabled() const {
  if (DoCounterPromotion.getNumOccurrences() > 0)
    return DoCounterPromotion;

  return Options.DoCounterPromotion;
}

void InstrProfiling::promoteCounterLoadStores(Function *F) {
  if (!isCounterPromotionEnabled())
    return;

  DominatorTree DT(*F);
  LoopInfo LI(DT);
  DenseMap<Loop *, SmallVector<LoadStorePair, 8>> LoopPromotionCandidates;

  std::unique_ptr<BlockFrequencyInfo> BFI;
  if (Options.UseBFIInPromotion) {
    std::unique_ptr<BranchProbabilityInfo> BPI;
    BPI.reset(new BranchProbabilityInfo(*F, LI, &GetTLI(*F)));
    BFI.reset(new BlockFrequencyInfo(*F, *BPI, LI));
  }

  for (const auto &LoadStore : PromotionCandidates) {
    auto *CounterLoad = LoadStore.first;
    auto *CounterStore = LoadStore.second;
    BasicBlock *BB = CounterLoad->getParent();
    Loop *ParentLoop = LI.getLoopFor(BB);
    if (!ParentLoop)
      continue;
    LoopPromotionCandidates[ParentLoop].emplace_back(CounterLoad, CounterStore);
  }

  SmallVector<Loop *, 4> Loops = LI.getLoopsInPreorder();

  // Do a post-order traversal of the loops so that counter updates can be
  // iteratively hoisted outside the loop nest.
  for (auto *Loop : llvm::reverse(Loops)) {
    PGOCounterPromoter Promoter(LoopPromotionCandidates, *Loop, LI, BFI.get());
    Promoter.run(&TotalCountersPromoted);
  }
}

static bool needsRuntimeHookUnconditionally(const Triple &TT) {
  // On Fuchsia, we only need runtime hook if any counters are present.
  if (TT.isOSFuchsia())
    return false;

  return true;
}

/// Check if the module contains uses of any profiling intrinsics.
static bool containsProfilingIntrinsics(Module &M) {
  auto containsIntrinsic = [&](int ID) {
    if (auto *F = M.getFunction(Intrinsic::getName(ID)))
      return !F->use_empty();
    return false;
  };
  return containsIntrinsic(llvm::Intrinsic::instrprof_cover) ||
         containsIntrinsic(llvm::Intrinsic::instrprof_increment) ||
         containsIntrinsic(llvm::Intrinsic::instrprof_increment_step) ||
         containsIntrinsic(llvm::Intrinsic::instrprof_value_profile);
}

bool InstrProfiling::run(
    Module &M, std::function<const TargetLibraryInfo &(Function &F)> GetTLI) {
  this->M = &M;
  this->GetTLI = std::move(GetTLI);
  NamesVar = nullptr;
  NamesSize = 0;
  ProfileDataMap.clear();
  CompilerUsedVars.clear();
  UsedVars.clear();
  TT = Triple(M.getTargetTriple());

  bool MadeChange = false;

  // Emit the runtime hook even if no counters are present.
  if (needsRuntimeHookUnconditionally(TT))
    MadeChange = emitRuntimeHook();

  // Improve compile time by avoiding linear scans when there is no work.
  GlobalVariable *CoverageNamesVar =
      M.getNamedGlobal(getCoverageUnusedNamesVarName());
  if (!containsProfilingIntrinsics(M) && !CoverageNamesVar)
    return MadeChange;

  // We did not know how many value sites there would be inside
  // the instrumented function. This is counting the number of instrumented
  // target value sites to enter it as field in the profile data variable.
  for (Function &F : M) {
    InstrProfIncrementInst *FirstProfIncInst = nullptr;
    for (BasicBlock &BB : F)
      for (auto I = BB.begin(), E = BB.end(); I != E; I++)
        if (auto *Ind = dyn_cast<InstrProfValueProfileInst>(I))
          computeNumValueSiteCounts(Ind);
        else if (FirstProfIncInst == nullptr)
          FirstProfIncInst = dyn_cast<InstrProfIncrementInst>(I);

    // Value profiling intrinsic lowering requires per-function profile data
    // variable to be created first.
    if (FirstProfIncInst != nullptr)
      static_cast<void>(getOrCreateRegionCounters(FirstProfIncInst));
  }

  for (Function &F : M)
    MadeChange |= lowerIntrinsics(&F);

  if (CoverageNamesVar) {
    lowerCoverageData(CoverageNamesVar);
    MadeChange = true;
  }

  if (!MadeChange)
    return false;

  emitVNodes();
  emitNameData();
  emitRuntimeHook();
  emitRegistration();
  emitUses();
  emitInitialization();
  return true;
}

static FunctionCallee getOrInsertValueProfilingCall(
    Module &M, const TargetLibraryInfo &TLI,
    ValueProfilingCallType CallType = ValueProfilingCallType::Default) {
  LLVMContext &Ctx = M.getContext();
  auto *ReturnTy = Type::getVoidTy(M.getContext());

  AttributeList AL;
  if (auto AK = TLI.getExtAttrForI32Param(false))
    AL = AL.addParamAttribute(M.getContext(), 2, AK);

  assert((CallType == ValueProfilingCallType::Default ||
          CallType == ValueProfilingCallType::MemOp) &&
         "Must be Default or MemOp");
  Type *ParamTypes[] = {
#define VALUE_PROF_FUNC_PARAM(ParamType, ParamName, ParamLLVMType) ParamLLVMType
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *ValueProfilingCallTy =
      FunctionType::get(ReturnTy, makeArrayRef(ParamTypes), false);
  StringRef FuncName = CallType == ValueProfilingCallType::Default
                           ? getInstrProfValueProfFuncName()
                           : getInstrProfValueProfMemOpFuncName();
  return M.getOrInsertFunction(FuncName, ValueProfilingCallTy, AL);
}

void InstrProfiling::computeNumValueSiteCounts(InstrProfValueProfileInst *Ind) {
  GlobalVariable *Name = Ind->getName();
  uint64_t ValueKind = Ind->getValueKind()->getZExtValue();
  uint64_t Index = Ind->getIndex()->getZExtValue();
  auto &PD = ProfileDataMap[Name];
  PD.NumValueSites[ValueKind] =
      std::max(PD.NumValueSites[ValueKind], (uint32_t)(Index + 1));
}

void InstrProfiling::lowerValueProfileInst(InstrProfValueProfileInst *Ind) {
  // TODO: Value profiling heavily depends on the data section which is omitted
  // in lightweight mode. We need to move the value profile pointer to the
  // Counter struct to get this working.
  assert(
      !DebugInfoCorrelate &&
      "Value profiling is not yet supported with lightweight instrumentation");
  GlobalVariable *Name = Ind->getName();
  auto It = ProfileDataMap.find(Name);
  assert(It != ProfileDataMap.end() && It->second.DataVar &&
         "value profiling detected in function with no counter incerement");

  GlobalVariable *DataVar = It->second.DataVar;
  uint64_t ValueKind = Ind->getValueKind()->getZExtValue();
  uint64_t Index = Ind->getIndex()->getZExtValue();
  for (uint32_t Kind = IPVK_First; Kind < ValueKind; ++Kind)
    Index += It->second.NumValueSites[Kind];

  IRBuilder<> Builder(Ind);
  bool IsMemOpSize = (Ind->getValueKind()->getZExtValue() ==
                      llvm::InstrProfValueKind::IPVK_MemOPSize);
  CallInst *Call = nullptr;
  auto *TLI = &GetTLI(*Ind->getFunction());

  // To support value profiling calls within Windows exception handlers, funclet
  // information contained within operand bundles needs to be copied over to
  // the library call. This is required for the IR to be processed by the
  // WinEHPrepare pass.
  SmallVector<OperandBundleDef, 1> OpBundles;
  Ind->getOperandBundlesAsDefs(OpBundles);
  if (!IsMemOpSize) {
    Value *Args[3] = {Ind->getTargetValue(),
                      Builder.CreateBitCast(DataVar, Builder.getInt8PtrTy()),
                      Builder.getInt32(Index)};
    Call = Builder.CreateCall(getOrInsertValueProfilingCall(*M, *TLI), Args,
                              OpBundles);
  } else {
    Value *Args[3] = {Ind->getTargetValue(),
                      Builder.CreateBitCast(DataVar, Builder.getInt8PtrTy()),
                      Builder.getInt32(Index)};
    Call = Builder.CreateCall(
        getOrInsertValueProfilingCall(*M, *TLI, ValueProfilingCallType::MemOp),
        Args, OpBundles);
  }
  if (auto AK = TLI->getExtAttrForI32Param(false))
    Call->addParamAttr(2, AK);
  Ind->replaceAllUsesWith(Call);
  Ind->eraseFromParent();
}

Value *InstrProfiling::getCounterAddress(InstrProfInstBase *I) {
  auto *Counters = getOrCreateRegionCounters(I);
  IRBuilder<> Builder(I);

  auto *Addr = Builder.CreateConstInBoundsGEP2_32(
      Counters->getValueType(), Counters, 0, I->getIndex()->getZExtValue());

  if (!isRuntimeCounterRelocationEnabled())
    return Addr;

  Type *Int64Ty = Type::getInt64Ty(M->getContext());
  Function *Fn = I->getParent()->getParent();
  LoadInst *&BiasLI = FunctionToProfileBiasMap[Fn];
  if (!BiasLI) {
    IRBuilder<> EntryBuilder(&Fn->getEntryBlock().front());
    auto *Bias = M->getGlobalVariable(getInstrProfCounterBiasVarName());
    if (!Bias) {
      // Compiler must define this variable when runtime counter relocation
      // is being used. Runtime has a weak external reference that is used
      // to check whether that's the case or not.
      Bias = new GlobalVariable(
          *M, Int64Ty, false, GlobalValue::LinkOnceODRLinkage,
          Constant::getNullValue(Int64Ty), getInstrProfCounterBiasVarName());
      Bias->setVisibility(GlobalVariable::HiddenVisibility);
      // A definition that's weak (linkonce_odr) without being in a COMDAT
      // section wouldn't lead to link errors, but it would lead to a dead
      // data word from every TU but one. Putting it in COMDAT ensures there
      // will be exactly one data slot in the link.
      if (TT.supportsCOMDAT())
        Bias->setComdat(M->getOrInsertComdat(Bias->getName()));
    }
    BiasLI = EntryBuilder.CreateLoad(Int64Ty, Bias);
  }
  auto *Add = Builder.CreateAdd(Builder.CreatePtrToInt(Addr, Int64Ty), BiasLI);
  return Builder.CreateIntToPtr(Add, Addr->getType());
}

void InstrProfiling::lowerCover(InstrProfCoverInst *CoverInstruction) {
  auto *Addr = getCounterAddress(CoverInstruction);
  IRBuilder<> Builder(CoverInstruction);
  // We store zero to represent that this block is covered.
  Builder.CreateStore(Builder.getInt8(0), Addr);
  CoverInstruction->eraseFromParent();
}

void InstrProfiling::lowerIncrement(InstrProfIncrementInst *Inc) {
  auto *Addr = getCounterAddress(Inc);

  IRBuilder<> Builder(Inc);
  if (Options.Atomic || AtomicCounterUpdateAll ||
      (Inc->getIndex()->isZeroValue() && AtomicFirstCounter)) {
    Builder.CreateAtomicRMW(AtomicRMWInst::Add, Addr, Inc->getStep(),
                            MaybeAlign(), AtomicOrdering::Monotonic);
  } else {
    Value *IncStep = Inc->getStep();
    Value *Load = Builder.CreateLoad(IncStep->getType(), Addr, "pgocount");
    auto *Count = Builder.CreateAdd(Load, Inc->getStep());
    auto *Store = Builder.CreateStore(Count, Addr);
    if (isCounterPromotionEnabled())
      PromotionCandidates.emplace_back(cast<Instruction>(Load), Store);
  }
  Inc->eraseFromParent();
}

void InstrProfiling::lowerCoverageData(GlobalVariable *CoverageNamesVar) {
  ConstantArray *Names =
      cast<ConstantArray>(CoverageNamesVar->getInitializer());
  for (unsigned I = 0, E = Names->getNumOperands(); I < E; ++I) {
    Constant *NC = Names->getOperand(I);
    Value *V = NC->stripPointerCasts();
    assert(isa<GlobalVariable>(V) && "Missing reference to function name");
    GlobalVariable *Name = cast<GlobalVariable>(V);

    Name->setLinkage(GlobalValue::PrivateLinkage);
    ReferencedNames.push_back(Name);
    if (isa<ConstantExpr>(NC))
      NC->dropAllReferences();
  }
  CoverageNamesVar->eraseFromParent();
}

/// Get the name of a profiling variable for a particular function.
static std::string getVarName(InstrProfInstBase *Inc, StringRef Prefix,
                              bool &Renamed) {
  StringRef NamePrefix = getInstrProfNameVarPrefix();
  StringRef Name = Inc->getName()->getName().substr(NamePrefix.size());
  Function *F = Inc->getParent()->getParent();
  Module *M = F->getParent();
  if (!DoHashBasedCounterSplit || !isIRPGOFlagSet(M) ||
      !canRenameComdatFunc(*F)) {
    Renamed = false;
    return (Prefix + Name).str();
  }
  Renamed = true;
  uint64_t FuncHash = Inc->getHash()->getZExtValue();
  SmallVector<char, 24> HashPostfix;
  if (Name.endswith((Twine(".") + Twine(FuncHash)).toStringRef(HashPostfix)))
    return (Prefix + Name).str();
  return (Prefix + Name + "." + Twine(FuncHash)).str();
}

static uint64_t getIntModuleFlagOrZero(const Module &M, StringRef Flag) {
  auto *MD = dyn_cast_or_null<ConstantAsMetadata>(M.getModuleFlag(Flag));
  if (!MD)
    return 0;

  // If the flag is a ConstantAsMetadata, it should be an integer representable
  // in 64-bits.
  return cast<ConstantInt>(MD->getValue())->getZExtValue();
}

static bool enablesValueProfiling(const Module &M) {
  return isIRPGOFlagSet(&M) ||
         getIntModuleFlagOrZero(M, "EnableValueProfiling") != 0;
}

// Conservatively returns true if data variables may be referenced by code.
static bool profDataReferencedByCode(const Module &M) {
  return enablesValueProfiling(M);
}

static inline bool shouldRecordFunctionAddr(Function *F) {
  // Only record function addresses if IR PGO is enabled or if clang value
  // profiling is enabled. Recording function addresses greatly increases object
  // file size, because it prevents the inliner from deleting functions that
  // have been inlined everywhere.
  if (!profDataReferencedByCode(*F->getParent()))
    return false;

  // Check the linkage
  bool HasAvailableExternallyLinkage = F->hasAvailableExternallyLinkage();
  if (!F->hasLinkOnceLinkage() && !F->hasLocalLinkage() &&
      !HasAvailableExternallyLinkage)
    return true;

  // A function marked 'alwaysinline' with available_externally linkage can't
  // have its address taken. Doing so would create an undefined external ref to
  // the function, which would fail to link.
  if (HasAvailableExternallyLinkage &&
      F->hasFnAttribute(Attribute::AlwaysInline))
    return false;

  // Prohibit function address recording if the function is both internal and
  // COMDAT. This avoids the profile data variable referencing internal symbols
  // in COMDAT.
  if (F->hasLocalLinkage() && F->hasComdat())
    return false;

  // Check uses of this function for other than direct calls or invokes to it.
  // Inline virtual functions have linkeOnceODR linkage. When a key method
  // exists, the vtable will only be emitted in the TU where the key method
  // is defined. In a TU where vtable is not available, the function won't
  // be 'addresstaken'. If its address is not recorded here, the profile data
  // with missing address may be picked by the linker leading  to missing
  // indirect call target info.
  return F->hasAddressTaken() || F->hasLinkOnceLinkage();
}

static bool needsRuntimeRegistrationOfSectionRange(const Triple &TT) {
  // Don't do this for Darwin.  compiler-rt uses linker magic.
  if (TT.isOSDarwin())
    return false;
  // Use linker script magic to get data/cnts/name start/end.
  if (TT.isOSAIX() || TT.isOSLinux() || TT.isOSFreeBSD() || TT.isOSNetBSD() ||
      TT.isOSSolaris() || TT.isOSFuchsia() || TT.isPS4() || TT.isOSWindows())
    return false;

  return true;
}

GlobalVariable *
InstrProfiling::createRegionCounters(InstrProfInstBase *Inc, StringRef Name,
                                     GlobalValue::LinkageTypes Linkage) {
  uint64_t NumCounters = Inc->getNumCounters()->getZExtValue();
  auto &Ctx = M->getContext();
  GlobalVariable *GV;
  if (isa<InstrProfCoverInst>(Inc)) {
    auto *CounterTy = Type::getInt8Ty(Ctx);
    auto *CounterArrTy = ArrayType::get(CounterTy, NumCounters);
    // TODO: `Constant::getAllOnesValue()` does not yet accept an array type.
    std::vector<Constant *> InitialValues(NumCounters,
                                          Constant::getAllOnesValue(CounterTy));
    GV = new GlobalVariable(*M, CounterArrTy, false, Linkage,
                            ConstantArray::get(CounterArrTy, InitialValues),
                            Name);
    GV->setAlignment(Align(1));
  } else {
    auto *CounterTy = ArrayType::get(Type::getInt64Ty(Ctx), NumCounters);
    GV = new GlobalVariable(*M, CounterTy, false, Linkage,
                            Constant::getNullValue(CounterTy), Name);
    GV->setAlignment(Align(8));
  }
  return GV;
}

GlobalVariable *
InstrProfiling::getOrCreateRegionCounters(InstrProfInstBase *Inc) {
  GlobalVariable *NamePtr = Inc->getName();
  auto &PD = ProfileDataMap[NamePtr];
  if (PD.RegionCounters)
    return PD.RegionCounters;

  // Match the linkage and visibility of the name global.
  Function *Fn = Inc->getParent()->getParent();
  GlobalValue::LinkageTypes Linkage = NamePtr->getLinkage();
  GlobalValue::VisibilityTypes Visibility = NamePtr->getVisibility();

  // Use internal rather than private linkage so the counter variable shows up
  // in the symbol table when using debug info for correlation.
  if (DebugInfoCorrelate && TT.isOSBinFormatMachO() &&
      Linkage == GlobalValue::PrivateLinkage)
    Linkage = GlobalValue::InternalLinkage;

  // Due to the limitation of binder as of 2021/09/28, the duplicate weak
  // symbols in the same csect won't be discarded. When there are duplicate weak
  // symbols, we can NOT guarantee that the relocations get resolved to the
  // intended weak symbol, so we can not ensure the correctness of the relative
  // CounterPtr, so we have to use private linkage for counter and data symbols.
  if (TT.isOSBinFormatXCOFF()) {
    Linkage = GlobalValue::PrivateLinkage;
    Visibility = GlobalValue::DefaultVisibility;
  }
  // Move the name variable to the right section. Place them in a COMDAT group
  // if the associated function is a COMDAT. This will make sure that only one
  // copy of counters of the COMDAT function will be emitted after linking. Keep
  // in mind that this pass may run before the inliner, so we need to create a
  // new comdat group for the counters and profiling data. If we use the comdat
  // of the parent function, that will result in relocations against discarded
  // sections.
  //
  // If the data variable is referenced by code,  counters and data have to be
  // in different comdats for COFF because the Visual C++ linker will report
  // duplicate symbol errors if there are multiple external symbols with the
  // same name marked IMAGE_COMDAT_SELECT_ASSOCIATIVE.
  //
  // For ELF, when not using COMDAT, put counters, data and values into a
  // nodeduplicate COMDAT which is lowered to a zero-flag section group. This
  // allows -z start-stop-gc to discard the entire group when the function is
  // discarded.
  bool DataReferencedByCode = profDataReferencedByCode(*M);
  bool NeedComdat = needsComdatForCounter(*Fn, *M);
  bool Renamed;
  std::string CntsVarName =
      getVarName(Inc, getInstrProfCountersVarPrefix(), Renamed);
  std::string DataVarName =
      getVarName(Inc, getInstrProfDataVarPrefix(), Renamed);
  auto MaybeSetComdat = [&](GlobalVariable *GV) {
    bool UseComdat = (NeedComdat || TT.isOSBinFormatELF());
    if (UseComdat) {
      StringRef GroupName = TT.isOSBinFormatCOFF() && DataReferencedByCode
                                ? GV->getName()
                                : CntsVarName;
      Comdat *C = M->getOrInsertComdat(GroupName);
      if (!NeedComdat)
        C->setSelectionKind(Comdat::NoDeduplicate);
      GV->setComdat(C);
    }
  };

  uint64_t NumCounters = Inc->getNumCounters()->getZExtValue();
  LLVMContext &Ctx = M->getContext();

  auto *CounterPtr = createRegionCounters(Inc, CntsVarName, Linkage);
  CounterPtr->setVisibility(Visibility);
  CounterPtr->setSection(
      getInstrProfSectionName(IPSK_cnts, TT.getObjectFormat()));
  MaybeSetComdat(CounterPtr);
  CounterPtr->setLinkage(Linkage);
  PD.RegionCounters = CounterPtr;
  if (DebugInfoCorrelate) {
    if (auto *SP = Fn->getSubprogram()) {
      DIBuilder DB(*M, true, SP->getUnit());
      Metadata *FunctionNameAnnotation[] = {
          MDString::get(Ctx, InstrProfCorrelator::FunctionNameAttributeName),
          MDString::get(Ctx, getPGOFuncNameVarInitializer(NamePtr)),
      };
      Metadata *CFGHashAnnotation[] = {
          MDString::get(Ctx, InstrProfCorrelator::CFGHashAttributeName),
          ConstantAsMetadata::get(Inc->getHash()),
      };
      Metadata *NumCountersAnnotation[] = {
          MDString::get(Ctx, InstrProfCorrelator::NumCountersAttributeName),
          ConstantAsMetadata::get(Inc->getNumCounters()),
      };
      auto Annotations = DB.getOrCreateArray({
          MDNode::get(Ctx, FunctionNameAnnotation),
          MDNode::get(Ctx, CFGHashAnnotation),
          MDNode::get(Ctx, NumCountersAnnotation),
      });
      auto *DICounter = DB.createGlobalVariableExpression(
          SP, CounterPtr->getName(), /*LinkageName=*/StringRef(), SP->getFile(),
          /*LineNo=*/0, DB.createUnspecifiedType("Profile Data Type"),
          CounterPtr->hasLocalLinkage(), /*IsDefined=*/true, /*Expr=*/nullptr,
          /*Decl=*/nullptr, /*TemplateParams=*/nullptr, /*AlignInBits=*/0,
          Annotations);
      CounterPtr->addDebugInfo(DICounter);
      DB.finalize();
    } else {
      std::string Msg = ("Missing debug info for function " + Fn->getName() +
                         "; required for profile correlation.")
                            .str();
      Ctx.diagnose(
          DiagnosticInfoPGOProfile(M->getName().data(), Msg, DS_Warning));
    }
  }

  auto *Int8PtrTy = Type::getInt8PtrTy(Ctx);
  // Allocate statically the array of pointers to value profile nodes for
  // the current function.
  Constant *ValuesPtrExpr = ConstantPointerNull::get(Int8PtrTy);
  uint64_t NS = 0;
  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
    NS += PD.NumValueSites[Kind];
  if (NS > 0 && ValueProfileStaticAlloc &&
      !needsRuntimeRegistrationOfSectionRange(TT)) {
    ArrayType *ValuesTy = ArrayType::get(Type::getInt64Ty(Ctx), NS);
    auto *ValuesVar = new GlobalVariable(
        *M, ValuesTy, false, Linkage, Constant::getNullValue(ValuesTy),
        getVarName(Inc, getInstrProfValuesVarPrefix(), Renamed));
    ValuesVar->setVisibility(Visibility);
    ValuesVar->setSection(
        getInstrProfSectionName(IPSK_vals, TT.getObjectFormat()));
    ValuesVar->setAlignment(Align(8));
    MaybeSetComdat(ValuesVar);
    ValuesPtrExpr =
        ConstantExpr::getBitCast(ValuesVar, Type::getInt8PtrTy(Ctx));
  }

  if (DebugInfoCorrelate) {
    // Mark the counter variable as used so that it isn't optimized out.
    CompilerUsedVars.push_back(PD.RegionCounters);
    return PD.RegionCounters;
  }

  // Create data variable.
  auto *IntPtrTy = M->getDataLayout().getIntPtrType(M->getContext());
  auto *Int16Ty = Type::getInt16Ty(Ctx);
  auto *Int16ArrayTy = ArrayType::get(Int16Ty, IPVK_Last + 1);
  Type *DataTypes[] = {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Init) LLVMType,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *DataTy = StructType::get(Ctx, makeArrayRef(DataTypes));

  Constant *FunctionAddr = shouldRecordFunctionAddr(Fn)
                               ? ConstantExpr::getBitCast(Fn, Int8PtrTy)
                               : ConstantPointerNull::get(Int8PtrTy);

  Constant *Int16ArrayVals[IPVK_Last + 1];
  for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
    Int16ArrayVals[Kind] = ConstantInt::get(Int16Ty, PD.NumValueSites[Kind]);

  // If the data variable is not referenced by code (if we don't emit
  // @llvm.instrprof.value.profile, NS will be 0), and the counter keeps the
  // data variable live under linker GC, the data variable can be private. This
  // optimization applies to ELF.
  //
  // On COFF, a comdat leader cannot be local so we require DataReferencedByCode
  // to be false.
  //
  // If profd is in a deduplicate comdat, NS==0 with a hash suffix guarantees
  // that other copies must have the same CFG and cannot have value profiling.
  // If no hash suffix, other profd copies may be referenced by code.
  if (NS == 0 && !(DataReferencedByCode && NeedComdat && !Renamed) &&
      (TT.isOSBinFormatELF() ||
       (!DataReferencedByCode && TT.isOSBinFormatCOFF()))) {
    Linkage = GlobalValue::PrivateLinkage;
    Visibility = GlobalValue::DefaultVisibility;
  }
  auto *Data =
      new GlobalVariable(*M, DataTy, false, Linkage, nullptr, DataVarName);
  // Reference the counter variable with a label difference (link-time
  // constant).
  auto *RelativeCounterPtr =
      ConstantExpr::getSub(ConstantExpr::getPtrToInt(CounterPtr, IntPtrTy),
                           ConstantExpr::getPtrToInt(Data, IntPtrTy));

  Constant *DataVals[] = {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Init) Init,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  Data->setInitializer(ConstantStruct::get(DataTy, DataVals));

  Data->setVisibility(Visibility);
  Data->setSection(getInstrProfSectionName(IPSK_data, TT.getObjectFormat()));
  Data->setAlignment(Align(INSTR_PROF_DATA_ALIGNMENT));
  MaybeSetComdat(Data);
  Data->setLinkage(Linkage);

  PD.DataVar = Data;

  // Mark the data variable as used so that it isn't stripped out.
  CompilerUsedVars.push_back(Data);
  // Now that the linkage set by the FE has been passed to the data and counter
  // variables, reset Name variable's linkage and visibility to private so that
  // it can be removed later by the compiler.
  NamePtr->setLinkage(GlobalValue::PrivateLinkage);
  // Collect the referenced names to be used by emitNameData.
  ReferencedNames.push_back(NamePtr);

  return PD.RegionCounters;
}

void InstrProfiling::emitVNodes() {
  if (!ValueProfileStaticAlloc)
    return;

  // For now only support this on platforms that do
  // not require runtime registration to discover
  // named section start/end.
  if (needsRuntimeRegistrationOfSectionRange(TT))
    return;

  size_t TotalNS = 0;
  for (auto &PD : ProfileDataMap) {
    for (uint32_t Kind = IPVK_First; Kind <= IPVK_Last; ++Kind)
      TotalNS += PD.second.NumValueSites[Kind];
  }

  if (!TotalNS)
    return;

  uint64_t NumCounters = TotalNS * NumCountersPerValueSite;
// Heuristic for small programs with very few total value sites.
// The default value of vp-counters-per-site is chosen based on
// the observation that large apps usually have a low percentage
// of value sites that actually have any profile data, and thus
// the average number of counters per site is low. For small
// apps with very few sites, this may not be true. Bump up the
// number of counters in this case.
#define INSTR_PROF_MIN_VAL_COUNTS 10
  if (NumCounters < INSTR_PROF_MIN_VAL_COUNTS)
    NumCounters = std::max(INSTR_PROF_MIN_VAL_COUNTS, (int)NumCounters * 2);

  auto &Ctx = M->getContext();
  Type *VNodeTypes[] = {
#define INSTR_PROF_VALUE_NODE(Type, LLVMType, Name, Init) LLVMType,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *VNodeTy = StructType::get(Ctx, makeArrayRef(VNodeTypes));

  ArrayType *VNodesTy = ArrayType::get(VNodeTy, NumCounters);
  auto *VNodesVar = new GlobalVariable(
      *M, VNodesTy, false, GlobalValue::PrivateLinkage,
      Constant::getNullValue(VNodesTy), getInstrProfVNodesVarName());
  VNodesVar->setSection(
      getInstrProfSectionName(IPSK_vnodes, TT.getObjectFormat()));
  // VNodesVar is used by runtime but not referenced via relocation by other
  // sections. Conservatively make it linker retained.
  UsedVars.push_back(VNodesVar);
}

void InstrProfiling::emitNameData() {
  std::string UncompressedData;

  if (ReferencedNames.empty())
    return;

  std::string CompressedNameStr;
  if (Error E = collectPGOFuncNameStrings(ReferencedNames, CompressedNameStr,
                                          DoInstrProfNameCompression)) {
    report_fatal_error(Twine(toString(std::move(E))), false);
  }

  auto &Ctx = M->getContext();
  auto *NamesVal =
      ConstantDataArray::getString(Ctx, StringRef(CompressedNameStr), false);
  NamesVar = new GlobalVariable(*M, NamesVal->getType(), true,
                                GlobalValue::PrivateLinkage, NamesVal,
                                getInstrProfNamesVarName());
  NamesSize = CompressedNameStr.size();
  NamesVar->setSection(
      getInstrProfSectionName(IPSK_name, TT.getObjectFormat()));
  // On COFF, it's important to reduce the alignment down to 1 to prevent the
  // linker from inserting padding before the start of the names section or
  // between names entries.
  NamesVar->setAlignment(Align(1));
  // NamesVar is used by runtime but not referenced via relocation by other
  // sections. Conservatively make it linker retained.
  UsedVars.push_back(NamesVar);

  for (auto *NamePtr : ReferencedNames)
    NamePtr->eraseFromParent();
}

void InstrProfiling::emitRegistration() {
  if (!needsRuntimeRegistrationOfSectionRange(TT))
    return;

  // Construct the function.
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *VoidPtrTy = Type::getInt8PtrTy(M->getContext());
  auto *Int64Ty = Type::getInt64Ty(M->getContext());
  auto *RegisterFTy = FunctionType::get(VoidTy, false);
  auto *RegisterF = Function::Create(RegisterFTy, GlobalValue::InternalLinkage,
                                     getInstrProfRegFuncsName(), M);
  RegisterF->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  if (Options.NoRedZone)
    RegisterF->addFnAttr(Attribute::NoRedZone);

  auto *RuntimeRegisterTy = FunctionType::get(VoidTy, VoidPtrTy, false);
  auto *RuntimeRegisterF =
      Function::Create(RuntimeRegisterTy, GlobalVariable::ExternalLinkage,
                       getInstrProfRegFuncName(), M);

  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", RegisterF));
  for (Value *Data : CompilerUsedVars)
    if (!isa<Function>(Data))
      IRB.CreateCall(RuntimeRegisterF, IRB.CreateBitCast(Data, VoidPtrTy));
  for (Value *Data : UsedVars)
    if (Data != NamesVar && !isa<Function>(Data))
      IRB.CreateCall(RuntimeRegisterF, IRB.CreateBitCast(Data, VoidPtrTy));

  if (NamesVar) {
    Type *ParamTypes[] = {VoidPtrTy, Int64Ty};
    auto *NamesRegisterTy =
        FunctionType::get(VoidTy, makeArrayRef(ParamTypes), false);
    auto *NamesRegisterF =
        Function::Create(NamesRegisterTy, GlobalVariable::ExternalLinkage,
                         getInstrProfNamesRegFuncName(), M);
    IRB.CreateCall(NamesRegisterF, {IRB.CreateBitCast(NamesVar, VoidPtrTy),
                                    IRB.getInt64(NamesSize)});
  }

  IRB.CreateRetVoid();
}

bool InstrProfiling::emitRuntimeHook() {
  // We expect the linker to be invoked with -u<hook_var> flag for Linux
  // in which case there is no need to emit the external variable.
  if (TT.isOSLinux())
    return false;

  // If the module's provided its own runtime, we don't need to do anything.
  if (M->getGlobalVariable(getInstrProfRuntimeHookVarName()))
    return false;

  // Declare an external variable that will pull in the runtime initialization.
  auto *Int32Ty = Type::getInt32Ty(M->getContext());
  auto *Var =
      new GlobalVariable(*M, Int32Ty, false, GlobalValue::ExternalLinkage,
                         nullptr, getInstrProfRuntimeHookVarName());

  if (TT.isOSBinFormatELF()) {
    // Mark the user variable as used so that it isn't stripped out.
    CompilerUsedVars.push_back(Var);
  } else {
    // Make a function that uses it.
    auto *User = Function::Create(FunctionType::get(Int32Ty, false),
                                  GlobalValue::LinkOnceODRLinkage,
                                  getInstrProfRuntimeHookVarUseFuncName(), M);
    User->addFnAttr(Attribute::NoInline);
    if (Options.NoRedZone)
      User->addFnAttr(Attribute::NoRedZone);
    User->setVisibility(GlobalValue::HiddenVisibility);
    if (TT.supportsCOMDAT())
      User->setComdat(M->getOrInsertComdat(User->getName()));

    IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", User));
    auto *Load = IRB.CreateLoad(Int32Ty, Var);
    IRB.CreateRet(Load);

    // Mark the function as used so that it isn't stripped out.
    CompilerUsedVars.push_back(User);
  }
  return true;
}

void InstrProfiling::emitUses() {
  // The metadata sections are parallel arrays. Optimizers (e.g.
  // GlobalOpt/ConstantMerge) may not discard associated sections as a unit, so
  // we conservatively retain all unconditionally in the compiler.
  //
  // On ELF and Mach-O, the linker can guarantee the associated sections will be
  // retained or discarded as a unit, so llvm.compiler.used is sufficient.
  // Similarly on COFF, if prof data is not referenced by code we use one comdat
  // and ensure this GC property as well. Otherwise, we have to conservatively
  // make all of the sections retained by the linker.
  if (TT.isOSBinFormatELF() || TT.isOSBinFormatMachO() ||
      (TT.isOSBinFormatCOFF() && !profDataReferencedByCode(*M)))
    appendToCompilerUsed(*M, CompilerUsedVars);
  else
    appendToUsed(*M, CompilerUsedVars);

  // We do not add proper references from used metadata sections to NamesVar and
  // VNodesVar, so we have to be conservative and place them in llvm.used
  // regardless of the target,
  appendToUsed(*M, UsedVars);
}

void InstrProfiling::emitInitialization() {
  // Create ProfileFileName variable. Don't don't this for the
  // context-sensitive instrumentation lowering: This lowering is after
  // LTO/ThinLTO linking. Pass PGOInstrumentationGenCreateVar should
  // have already create the variable before LTO/ThinLTO linking.
  if (!IsCS)
    createProfileFileNameVar(*M, Options.InstrProfileOutput);
  Function *RegisterF = M->getFunction(getInstrProfRegFuncsName());
  if (!RegisterF)
    return;

  // Create the initialization function.
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *F = Function::Create(FunctionType::get(VoidTy, false),
                             GlobalValue::InternalLinkage,
                             getInstrProfInitFuncName(), M);
  F->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  F->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone)
    F->addFnAttr(Attribute::NoRedZone);

  // Add the basic block and the necessary calls.
  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", F));
  IRB.CreateCall(RegisterF, {});
  IRB.CreateRetVoid();

  appendToGlobalCtors(*M, F, 0);
}
