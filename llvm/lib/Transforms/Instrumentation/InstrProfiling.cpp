//===-- InstrProfiling.cpp - Frontend instrumentation based profiling -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers instrprof_* intrinsics emitted by an instrumentor.
// It also builds the data structures and initialization code needed for
// updating execution counts and emitting the profile at runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
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
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "instrprof"

namespace llvm {
// TODO: Remove -debug-info-correlate in next LLVM release, in favor of
// -profile-correlate=debug-info.
cl::opt<bool> DebugInfoCorrelate(
    "debug-info-correlate",
    cl::desc("Use debug info to correlate profiles. (Deprecated, use "
             "-profile-correlate=debug-info)"),
    cl::init(false));

cl::opt<InstrProfCorrelator::ProfCorrelatorKind> ProfileCorrelate(
    "profile-correlate",
    cl::desc("Use debug info or binary file to correlate profiles."),
    cl::init(InstrProfCorrelator::NONE),
    cl::values(clEnumValN(InstrProfCorrelator::NONE, "",
                          "No profile correlation"),
               clEnumValN(InstrProfCorrelator::DEBUG_INFO, "debug-info",
                          "Use debug info to correlate"),
               clEnumValN(InstrProfCorrelator::BINARY, "binary",
                          "Use binary to correlate")));
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
    "max-counter-promotions-per-loop", cl::init(20),
    cl::desc("Max number counter promotions per loop to avoid"
             " increasing register pressure too much"));

// A debug option
cl::opt<int>
    MaxNumOfPromotions("max-counter-promotions", cl::init(-1),
                       cl::desc("Max number of allowed counter promotions"));

cl::opt<unsigned> SpeculativeCounterPromotionMaxExiting(
    "speculative-counter-promotion-max-exiting", cl::init(3),
    cl::desc("The max number of exiting blocks of a loop to allow "
             " speculative counter promotion"));

cl::opt<bool> SpeculativeCounterPromotionToLoop(
    "speculative-counter-promotion-to-loop",
    cl::desc("When the option is false, if the target block is in a loop, "
             "the promotion will be disallowed unless the promoted counter "
             " update can be further/iteratively promoted into an acyclic "
             " region."));

cl::opt<bool> IterativeCounterPromotion(
    "iterative-counter-promotion", cl::init(true),
    cl::desc("Allow counter promotion across the whole loop nest."));

cl::opt<bool> SkipRetExitBlock(
    "skip-ret-exit-block", cl::init(true),
    cl::desc("Suppress counter promotion if exit blocks contain ret."));

using LoadStorePair = std::pair<Instruction *, Instruction *>;

class InstrLowerer final {
public:
  InstrLowerer(Module &M, const InstrProfOptions &Options,
               std::function<const TargetLibraryInfo &(Function &F)> GetTLI,
               bool IsCS)
      : M(M), Options(Options), TT(Triple(M.getTargetTriple())), IsCS(IsCS),
        GetTLI(GetTLI) {}

  bool lower();

private:
  Module &M;
  const InstrProfOptions Options;
  const Triple TT;
  // Is this lowering for the context-sensitive instrumentation.
  const bool IsCS;

  std::function<const TargetLibraryInfo &(Function &F)> GetTLI;
  struct PerFunctionProfileData {
    uint32_t NumValueSites[IPVK_Last + 1] = {};
    GlobalVariable *RegionCounters = nullptr;
    GlobalVariable *DataVar = nullptr;
    GlobalVariable *RegionBitmaps = nullptr;
    uint32_t NumBitmapBytes = 0;

    PerFunctionProfileData() = default;
  };
  DenseMap<GlobalVariable *, PerFunctionProfileData> ProfileDataMap;
  /// If runtime relocation is enabled, this maps functions to the load
  /// instruction that produces the profile relocation bias.
  DenseMap<const Function *, LoadInst *> FunctionToProfileBiasMap;
  std::vector<GlobalValue *> CompilerUsedVars;
  std::vector<GlobalValue *> UsedVars;
  std::vector<GlobalVariable *> ReferencedNames;
  GlobalVariable *NamesVar = nullptr;
  size_t NamesSize = 0;

  // vector of counter load/store pairs to be register promoted.
  std::vector<LoadStorePair> PromotionCandidates;

  int64_t TotalCountersPromoted = 0;

  /// Lower instrumentation intrinsics in the function. Returns true if there
  /// any lowering.
  bool lowerIntrinsics(Function *F);

  /// Register-promote counter loads and stores in loops.
  void promoteCounterLoadStores(Function *F);

  /// Returns true if relocating counters at runtime is enabled.
  bool isRuntimeCounterRelocationEnabled() const;

  /// Returns true if profile counter update register promotion is enabled.
  bool isCounterPromotionEnabled() const;

  /// Count the number of instrumented value sites for the function.
  void computeNumValueSiteCounts(InstrProfValueProfileInst *Ins);

  /// Replace instrprof.value.profile with a call to runtime library.
  void lowerValueProfileInst(InstrProfValueProfileInst *Ins);

  /// Replace instrprof.cover with a store instruction to the coverage byte.
  void lowerCover(InstrProfCoverInst *Inc);

  /// Replace instrprof.timestamp with a call to
  /// INSTR_PROF_PROFILE_SET_TIMESTAMP.
  void lowerTimestamp(InstrProfTimestampInst *TimestampInstruction);

  /// Replace instrprof.increment with an increment of the appropriate value.
  void lowerIncrement(InstrProfIncrementInst *Inc);

  /// Force emitting of name vars for unused functions.
  void lowerCoverageData(GlobalVariable *CoverageNamesVar);

  /// Replace instrprof.mcdc.tvbitmask.update with a shift and or instruction
  /// using the index represented by the a temp value into a bitmap.
  void lowerMCDCTestVectorBitmapUpdate(InstrProfMCDCTVBitmapUpdate *Ins);

  /// Replace instrprof.mcdc.temp.update with a shift and or instruction using
  /// the corresponding condition ID.
  void lowerMCDCCondBitmapUpdate(InstrProfMCDCCondBitmapUpdate *Ins);

  /// Compute the address of the counter value that this profiling instruction
  /// acts on.
  Value *getCounterAddress(InstrProfCntrInstBase *I);

  /// Get the region counters for an increment, creating them if necessary.
  ///
  /// If the counter array doesn't yet exist, the profile data variables
  /// referring to them will also be created.
  GlobalVariable *getOrCreateRegionCounters(InstrProfCntrInstBase *Inc);

  /// Create the region counters.
  GlobalVariable *createRegionCounters(InstrProfCntrInstBase *Inc,
                                       StringRef Name,
                                       GlobalValue::LinkageTypes Linkage);

  /// Compute the address of the test vector bitmap that this profiling
  /// instruction acts on.
  Value *getBitmapAddress(InstrProfMCDCTVBitmapUpdate *I);

  /// Get the region bitmaps for an increment, creating them if necessary.
  ///
  /// If the bitmap array doesn't yet exist, the profile data variables
  /// referring to them will also be created.
  GlobalVariable *getOrCreateRegionBitmaps(InstrProfMCDCBitmapInstBase *Inc);

  /// Create the MC/DC bitmap as a byte-aligned array of bytes associated with
  /// an MC/DC Decision region. The number of bytes required is indicated by
  /// the intrinsic used (type InstrProfMCDCBitmapInstBase).  This is called
  /// as part of setupProfileSection() and is conceptually very similar to
  /// what is done for profile data counters in createRegionCounters().
  GlobalVariable *createRegionBitmaps(InstrProfMCDCBitmapInstBase *Inc,
                                      StringRef Name,
                                      GlobalValue::LinkageTypes Linkage);

  /// Set Comdat property of GV, if required.
  void maybeSetComdat(GlobalVariable *GV, Function *Fn, StringRef VarName);

  /// Setup the sections into which counters and bitmaps are allocated.
  GlobalVariable *setupProfileSection(InstrProfInstBase *Inc,
                                      InstrProfSectKind IPSK);

  /// Create INSTR_PROF_DATA variable for counters and bitmaps.
  void createDataVariable(InstrProfCntrInstBase *Inc);

  /// Emit the section with compressed function names.
  void emitNameData();

  /// Emit value nodes section for value profiling.
  void emitVNodes();

  /// Emit runtime registration functions for each profile data variable.
  void emitRegistration();

  /// Emit the necessary plumbing to pull in the runtime initialization.
  /// Returns true if a change was made.
  bool emitRuntimeHook();

  /// Add uses of our data variables and runtime hook.
  void emitUses();

  /// Create a static initializer for our data, on platforms that need it,
  /// and for any profile output file that was specified.
  void emitInitialization();
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
        Addr = Builder.CreateIntToPtr(BiasInst,
                                      PointerType::getUnqual(Ty->getContext()));
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
      if (BlockSet.insert(ExitBlock).second &&
          llvm::none_of(predecessors(ExitBlock), [&](const BasicBlock *Pred) {
            return llvm::isPresplitCoroSuspendExitEdge(*Pred, *ExitBlock);
          })) {
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
      for (auto *BB : ExitBlocks)
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
        if (PreheaderCount && (*PreheaderCount * 3) >= (*InstrCount * 2))
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

PreservedAnalyses InstrProfilingLoweringPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  InstrLowerer Lowerer(M, Options, GetTLI, IsCS);
  if (!Lowerer.lower())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

bool InstrLowerer::lowerIntrinsics(Function *F) {
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
      } else if (auto *IPC = dyn_cast<InstrProfTimestampInst>(&Instr)) {
        lowerTimestamp(IPC);
        MadeChange = true;
      } else if (auto *IPC = dyn_cast<InstrProfCoverInst>(&Instr)) {
        lowerCover(IPC);
        MadeChange = true;
      } else if (auto *IPVP = dyn_cast<InstrProfValueProfileInst>(&Instr)) {
        lowerValueProfileInst(IPVP);
        MadeChange = true;
      } else if (auto *IPMP = dyn_cast<InstrProfMCDCBitmapParameters>(&Instr)) {
        IPMP->eraseFromParent();
        MadeChange = true;
      } else if (auto *IPBU = dyn_cast<InstrProfMCDCTVBitmapUpdate>(&Instr)) {
        lowerMCDCTestVectorBitmapUpdate(IPBU);
        MadeChange = true;
      } else if (auto *IPTU = dyn_cast<InstrProfMCDCCondBitmapUpdate>(&Instr)) {
        lowerMCDCCondBitmapUpdate(IPTU);
        MadeChange = true;
      }
    }
  }

  if (!MadeChange)
    return false;

  promoteCounterLoadStores(F);
  return true;
}

bool InstrLowerer::isRuntimeCounterRelocationEnabled() const {
  // Mach-O don't support weak external references.
  if (TT.isOSBinFormatMachO())
    return false;

  if (RuntimeCounterRelocation.getNumOccurrences() > 0)
    return RuntimeCounterRelocation;

  // Fuchsia uses runtime counter relocation by default.
  return TT.isOSFuchsia();
}

bool InstrLowerer::isCounterPromotionEnabled() const {
  if (DoCounterPromotion.getNumOccurrences() > 0)
    return DoCounterPromotion;

  return Options.DoCounterPromotion;
}

void InstrLowerer::promoteCounterLoadStores(Function *F) {
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
         containsIntrinsic(llvm::Intrinsic::instrprof_timestamp) ||
         containsIntrinsic(llvm::Intrinsic::instrprof_value_profile);
}

bool InstrLowerer::lower() {
  bool MadeChange = false;
  bool NeedsRuntimeHook = needsRuntimeHookUnconditionally(TT);
  if (NeedsRuntimeHook)
    MadeChange = emitRuntimeHook();

  bool ContainsProfiling = containsProfilingIntrinsics(M);
  GlobalVariable *CoverageNamesVar =
      M.getNamedGlobal(getCoverageUnusedNamesVarName());
  // Improve compile time by avoiding linear scans when there is no work.
  if (!ContainsProfiling && !CoverageNamesVar)
    return MadeChange;

  // We did not know how many value sites there would be inside
  // the instrumented function. This is counting the number of instrumented
  // target value sites to enter it as field in the profile data variable.
  for (Function &F : M) {
    InstrProfCntrInstBase *FirstProfInst = nullptr;
    for (BasicBlock &BB : F) {
      for (auto I = BB.begin(), E = BB.end(); I != E; I++) {
        if (auto *Ind = dyn_cast<InstrProfValueProfileInst>(I))
          computeNumValueSiteCounts(Ind);
        else {
          if (FirstProfInst == nullptr &&
              (isa<InstrProfIncrementInst>(I) || isa<InstrProfCoverInst>(I)))
            FirstProfInst = dyn_cast<InstrProfCntrInstBase>(I);
          // If the MCDCBitmapParameters intrinsic seen, create the bitmaps.
          if (const auto &Params = dyn_cast<InstrProfMCDCBitmapParameters>(I))
            static_cast<void>(getOrCreateRegionBitmaps(Params));
        }
      }
    }

    // Use a profile intrinsic to create the region counters and data variable.
    // Also create the data variable based on the MCDCParams.
    if (FirstProfInst != nullptr) {
      static_cast<void>(getOrCreateRegionCounters(FirstProfInst));
    }
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

  // Emit runtime hook for the cases where the target does not unconditionally
  // require pulling in profile runtime, and coverage is enabled on code that is
  // not eliminated by the front-end, e.g. unused functions with internal
  // linkage.
  if (!NeedsRuntimeHook && ContainsProfiling)
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
      FunctionType::get(ReturnTy, ArrayRef(ParamTypes), false);
  StringRef FuncName = CallType == ValueProfilingCallType::Default
                           ? getInstrProfValueProfFuncName()
                           : getInstrProfValueProfMemOpFuncName();
  return M.getOrInsertFunction(FuncName, ValueProfilingCallTy, AL);
}

void InstrLowerer::computeNumValueSiteCounts(InstrProfValueProfileInst *Ind) {
  GlobalVariable *Name = Ind->getName();
  uint64_t ValueKind = Ind->getValueKind()->getZExtValue();
  uint64_t Index = Ind->getIndex()->getZExtValue();
  auto &PD = ProfileDataMap[Name];
  PD.NumValueSites[ValueKind] =
      std::max(PD.NumValueSites[ValueKind], (uint32_t)(Index + 1));
}

void InstrLowerer::lowerValueProfileInst(InstrProfValueProfileInst *Ind) {
  // TODO: Value profiling heavily depends on the data section which is omitted
  // in lightweight mode. We need to move the value profile pointer to the
  // Counter struct to get this working.
  assert(
      !DebugInfoCorrelate && ProfileCorrelate == InstrProfCorrelator::NONE &&
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
    Value *Args[3] = {Ind->getTargetValue(), DataVar, Builder.getInt32(Index)};
    Call = Builder.CreateCall(getOrInsertValueProfilingCall(M, *TLI), Args,
                              OpBundles);
  } else {
    Value *Args[3] = {Ind->getTargetValue(), DataVar, Builder.getInt32(Index)};
    Call = Builder.CreateCall(
        getOrInsertValueProfilingCall(M, *TLI, ValueProfilingCallType::MemOp),
        Args, OpBundles);
  }
  if (auto AK = TLI->getExtAttrForI32Param(false))
    Call->addParamAttr(2, AK);
  Ind->replaceAllUsesWith(Call);
  Ind->eraseFromParent();
}

Value *InstrLowerer::getCounterAddress(InstrProfCntrInstBase *I) {
  auto *Counters = getOrCreateRegionCounters(I);
  IRBuilder<> Builder(I);

  if (isa<InstrProfTimestampInst>(I))
    Counters->setAlignment(Align(8));

  auto *Addr = Builder.CreateConstInBoundsGEP2_32(
      Counters->getValueType(), Counters, 0, I->getIndex()->getZExtValue());

  if (!isRuntimeCounterRelocationEnabled())
    return Addr;

  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  Function *Fn = I->getParent()->getParent();
  LoadInst *&BiasLI = FunctionToProfileBiasMap[Fn];
  if (!BiasLI) {
    IRBuilder<> EntryBuilder(&Fn->getEntryBlock().front());
    auto *Bias = M.getGlobalVariable(getInstrProfCounterBiasVarName());
    if (!Bias) {
      // Compiler must define this variable when runtime counter relocation
      // is being used. Runtime has a weak external reference that is used
      // to check whether that's the case or not.
      Bias = new GlobalVariable(
          M, Int64Ty, false, GlobalValue::LinkOnceODRLinkage,
          Constant::getNullValue(Int64Ty), getInstrProfCounterBiasVarName());
      Bias->setVisibility(GlobalVariable::HiddenVisibility);
      // A definition that's weak (linkonce_odr) without being in a COMDAT
      // section wouldn't lead to link errors, but it would lead to a dead
      // data word from every TU but one. Putting it in COMDAT ensures there
      // will be exactly one data slot in the link.
      if (TT.supportsCOMDAT())
        Bias->setComdat(M.getOrInsertComdat(Bias->getName()));
    }
    BiasLI = EntryBuilder.CreateLoad(Int64Ty, Bias);
  }
  auto *Add = Builder.CreateAdd(Builder.CreatePtrToInt(Addr, Int64Ty), BiasLI);
  return Builder.CreateIntToPtr(Add, Addr->getType());
}

Value *InstrLowerer::getBitmapAddress(InstrProfMCDCTVBitmapUpdate *I) {
  auto *Bitmaps = getOrCreateRegionBitmaps(I);
  IRBuilder<> Builder(I);

  auto *Addr = Builder.CreateConstInBoundsGEP2_32(
      Bitmaps->getValueType(), Bitmaps, 0, I->getBitmapIndex()->getZExtValue());

  if (isRuntimeCounterRelocationEnabled()) {
    LLVMContext &Ctx = M.getContext();
    Ctx.diagnose(DiagnosticInfoPGOProfile(
        M.getName().data(),
        Twine("Runtime counter relocation is presently not supported for MC/DC "
              "bitmaps."),
        DS_Warning));
  }

  return Addr;
}

void InstrLowerer::lowerCover(InstrProfCoverInst *CoverInstruction) {
  auto *Addr = getCounterAddress(CoverInstruction);
  IRBuilder<> Builder(CoverInstruction);
  // We store zero to represent that this block is covered.
  Builder.CreateStore(Builder.getInt8(0), Addr);
  CoverInstruction->eraseFromParent();
}

void InstrLowerer::lowerTimestamp(
    InstrProfTimestampInst *TimestampInstruction) {
  assert(TimestampInstruction->getIndex()->isZeroValue() &&
         "timestamp probes are always the first probe for a function");
  auto &Ctx = M.getContext();
  auto *TimestampAddr = getCounterAddress(TimestampInstruction);
  IRBuilder<> Builder(TimestampInstruction);
  auto *CalleeTy =
      FunctionType::get(Type::getVoidTy(Ctx), TimestampAddr->getType(), false);
  auto Callee = M.getOrInsertFunction(
      INSTR_PROF_QUOTE(INSTR_PROF_PROFILE_SET_TIMESTAMP), CalleeTy);
  Builder.CreateCall(Callee, {TimestampAddr});
  TimestampInstruction->eraseFromParent();
}

void InstrLowerer::lowerIncrement(InstrProfIncrementInst *Inc) {
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

void InstrLowerer::lowerCoverageData(GlobalVariable *CoverageNamesVar) {
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

void InstrLowerer::lowerMCDCTestVectorBitmapUpdate(
    InstrProfMCDCTVBitmapUpdate *Update) {
  IRBuilder<> Builder(Update);
  auto *Int8Ty = Type::getInt8Ty(M.getContext());
  auto *Int8PtrTy = PointerType::getUnqual(M.getContext());
  auto *Int32Ty = Type::getInt32Ty(M.getContext());
  auto *Int64Ty = Type::getInt64Ty(M.getContext());
  auto *MCDCCondBitmapAddr = Update->getMCDCCondBitmapAddr();
  auto *BitmapAddr = getBitmapAddress(Update);

  // Load Temp Val.
  //  %mcdc.temp = load i32, ptr %mcdc.addr, align 4
  auto *Temp = Builder.CreateLoad(Int32Ty, MCDCCondBitmapAddr, "mcdc.temp");

  // Calculate byte offset using div8.
  //  %1 = lshr i32 %mcdc.temp, 3
  auto *BitmapByteOffset = Builder.CreateLShr(Temp, 0x3);

  // Add byte offset to section base byte address.
  //  %2 = zext i32 %1 to i64
  //  %3 = add i64 ptrtoint (ptr @__profbm_test to i64), %2
  auto *BitmapByteAddr =
      Builder.CreateAdd(Builder.CreatePtrToInt(BitmapAddr, Int64Ty),
                        Builder.CreateZExtOrBitCast(BitmapByteOffset, Int64Ty));

  // Convert to a pointer.
  //  %4 = inttoptr i32 %3 to ptr
  BitmapByteAddr = Builder.CreateIntToPtr(BitmapByteAddr, Int8PtrTy);

  // Calculate bit offset into bitmap byte by using div8 remainder (AND ~8)
  //  %5 = and i32 %mcdc.temp, 7
  //  %6 = trunc i32 %5 to i8
  auto *BitToSet = Builder.CreateTrunc(Builder.CreateAnd(Temp, 0x7), Int8Ty);

  // Shift bit offset left to form a bitmap.
  //  %7 = shl i8 1, %6
  auto *ShiftedVal = Builder.CreateShl(Builder.getInt8(0x1), BitToSet);

  // Load profile bitmap byte.
  //  %mcdc.bits = load i8, ptr %4, align 1
  auto *Bitmap = Builder.CreateLoad(Int8Ty, BitmapByteAddr, "mcdc.bits");

  // Perform logical OR of profile bitmap byte and shifted bit offset.
  //  %8 = or i8 %mcdc.bits, %7
  auto *Result = Builder.CreateOr(Bitmap, ShiftedVal);

  // Store the updated profile bitmap byte.
  //  store i8 %8, ptr %3, align 1
  Builder.CreateStore(Result, BitmapByteAddr);
  Update->eraseFromParent();
}

void InstrLowerer::lowerMCDCCondBitmapUpdate(
    InstrProfMCDCCondBitmapUpdate *Update) {
  IRBuilder<> Builder(Update);
  auto *Int32Ty = Type::getInt32Ty(M.getContext());
  auto *MCDCCondBitmapAddr = Update->getMCDCCondBitmapAddr();

  // Load the MCDC temporary value from the stack.
  //  %mcdc.temp = load i32, ptr %mcdc.addr, align 4
  auto *Temp = Builder.CreateLoad(Int32Ty, MCDCCondBitmapAddr, "mcdc.temp");

  // Zero-extend the evaluated condition boolean value (0 or 1) by 32bits.
  //  %1 = zext i1 %tobool to i32
  auto *CondV_32 = Builder.CreateZExt(Update->getCondBool(), Int32Ty);

  // Shift the boolean value left (by the condition's ID) to form a bitmap.
  //  %2 = shl i32 %1, <Update->getCondID()>
  auto *ShiftedVal = Builder.CreateShl(CondV_32, Update->getCondID());

  // Perform logical OR of the bitmap against the loaded MCDC temporary value.
  //  %3 = or i32 %mcdc.temp, %2
  auto *Result = Builder.CreateOr(Temp, ShiftedVal);

  // Store the updated temporary value back to the stack.
  //  store i32 %3, ptr %mcdc.addr, align 4
  Builder.CreateStore(Result, MCDCCondBitmapAddr);
  Update->eraseFromParent();
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
  if (Name.ends_with((Twine(".") + Twine(FuncHash)).toStringRef(HashPostfix)))
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

static inline bool shouldUsePublicSymbol(Function *Fn) {
  // It isn't legal to make an alias of this function at all
  if (Fn->isDeclarationForLinker())
    return true;

  // Symbols with local linkage can just use the symbol directly without
  // introducing relocations
  if (Fn->hasLocalLinkage())
    return true;

  // PGO + ThinLTO + CFI cause duplicate symbols to be introduced due to some
  // unfavorable interaction between the new alias and the alias renaming done
  // in LowerTypeTests under ThinLTO. For comdat functions that would normally
  // be deduplicated, but the renaming scheme ends up preventing renaming, since
  // it creates unique names for each alias, resulting in duplicated symbols. In
  // the future, we should update the CFI related passes to migrate these
  // aliases to the same module as the jump-table they refer to will be defined.
  if (Fn->hasMetadata(LLVMContext::MD_type))
    return true;

  // For comdat functions, an alias would need the same linkage as the original
  // function and hidden visibility. There is no point in adding an alias with
  // identical linkage an visibility to avoid introducing symbolic relocations.
  if (Fn->hasComdat() &&
      (Fn->getVisibility() == GlobalValue::VisibilityTypes::HiddenVisibility))
    return true;

  // its OK to use an alias
  return false;
}

static inline Constant *getFuncAddrForProfData(Function *Fn) {
  auto *Int8PtrTy = PointerType::getUnqual(Fn->getContext());
  // Store a nullptr in __llvm_profd, if we shouldn't use a real address
  if (!shouldRecordFunctionAddr(Fn))
    return ConstantPointerNull::get(Int8PtrTy);

  // If we can't use an alias, we must use the public symbol, even though this
  // may require a symbolic relocation.
  if (shouldUsePublicSymbol(Fn))
    return Fn;

  // When possible use a private alias to avoid symbolic relocations.
  auto *GA = GlobalAlias::create(GlobalValue::LinkageTypes::PrivateLinkage,
                                 Fn->getName() + ".local", Fn);

  // When the instrumented function is a COMDAT function, we cannot use a
  // private alias. If we did, we would create reference to a local label in
  // this function's section. If this version of the function isn't selected by
  // the linker, then the metadata would introduce a reference to a discarded
  // section. So, for COMDAT functions, we need to adjust the linkage of the
  // alias. Using hidden visibility avoids a dynamic relocation and an entry in
  // the dynamic symbol table.
  //
  // Note that this handles COMDAT functions with visibility other than Hidden,
  // since that case is covered in shouldUsePublicSymbol()
  if (Fn->hasComdat()) {
    GA->setLinkage(Fn->getLinkage());
    GA->setVisibility(GlobalValue::VisibilityTypes::HiddenVisibility);
  }

  // appendToCompilerUsed(*Fn->getParent(), {GA});

  return GA;
}

static bool needsRuntimeRegistrationOfSectionRange(const Triple &TT) {
  // compiler-rt uses linker support to get data/counters/name start/end for
  // ELF, COFF, Mach-O and XCOFF.
  if (TT.isOSBinFormatELF() || TT.isOSBinFormatCOFF() ||
      TT.isOSBinFormatMachO() || TT.isOSBinFormatXCOFF())
    return false;

  return true;
}

void InstrLowerer::maybeSetComdat(GlobalVariable *GV, Function *Fn,
                                  StringRef VarName) {
  bool DataReferencedByCode = profDataReferencedByCode(M);
  bool NeedComdat = needsComdatForCounter(*Fn, M);
  bool UseComdat = (NeedComdat || TT.isOSBinFormatELF());

  if (!UseComdat)
    return;

  StringRef GroupName =
      TT.isOSBinFormatCOFF() && DataReferencedByCode ? GV->getName() : VarName;
  Comdat *C = M.getOrInsertComdat(GroupName);
  if (!NeedComdat)
    C->setSelectionKind(Comdat::NoDeduplicate);
  GV->setComdat(C);
  // COFF doesn't allow the comdat group leader to have private linkage, so
  // upgrade private linkage to internal linkage to produce a symbol table
  // entry.
  if (TT.isOSBinFormatCOFF() && GV->hasPrivateLinkage())
    GV->setLinkage(GlobalValue::InternalLinkage);
}

GlobalVariable *InstrLowerer::setupProfileSection(InstrProfInstBase *Inc,
                                                  InstrProfSectKind IPSK) {
  GlobalVariable *NamePtr = Inc->getName();

  // Match the linkage and visibility of the name global.
  Function *Fn = Inc->getParent()->getParent();
  GlobalValue::LinkageTypes Linkage = NamePtr->getLinkage();
  GlobalValue::VisibilityTypes Visibility = NamePtr->getVisibility();

  // Use internal rather than private linkage so the counter variable shows up
  // in the symbol table when using debug info for correlation.
  if ((DebugInfoCorrelate ||
       ProfileCorrelate == InstrProfCorrelator::DEBUG_INFO) &&
      TT.isOSBinFormatMachO() && Linkage == GlobalValue::PrivateLinkage)
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
  bool Renamed;
  GlobalVariable *Ptr;
  StringRef VarPrefix;
  std::string VarName;
  if (IPSK == IPSK_cnts) {
    VarPrefix = getInstrProfCountersVarPrefix();
    VarName = getVarName(Inc, VarPrefix, Renamed);
    InstrProfCntrInstBase *CntrIncrement = dyn_cast<InstrProfCntrInstBase>(Inc);
    Ptr = createRegionCounters(CntrIncrement, VarName, Linkage);
  } else if (IPSK == IPSK_bitmap) {
    VarPrefix = getInstrProfBitmapVarPrefix();
    VarName = getVarName(Inc, VarPrefix, Renamed);
    InstrProfMCDCBitmapInstBase *BitmapUpdate =
        dyn_cast<InstrProfMCDCBitmapInstBase>(Inc);
    Ptr = createRegionBitmaps(BitmapUpdate, VarName, Linkage);
  } else {
    llvm_unreachable("Profile Section must be for Counters or Bitmaps");
  }

  Ptr->setVisibility(Visibility);
  // Put the counters and bitmaps in their own sections so linkers can
  // remove unneeded sections.
  Ptr->setSection(getInstrProfSectionName(IPSK, TT.getObjectFormat()));
  Ptr->setLinkage(Linkage);
  maybeSetComdat(Ptr, Fn, VarName);
  return Ptr;
}

GlobalVariable *
InstrLowerer::createRegionBitmaps(InstrProfMCDCBitmapInstBase *Inc,
                                  StringRef Name,
                                  GlobalValue::LinkageTypes Linkage) {
  uint64_t NumBytes = Inc->getNumBitmapBytes()->getZExtValue();
  auto *BitmapTy = ArrayType::get(Type::getInt8Ty(M.getContext()), NumBytes);
  auto GV = new GlobalVariable(M, BitmapTy, false, Linkage,
                               Constant::getNullValue(BitmapTy), Name);
  GV->setAlignment(Align(1));
  return GV;
}

GlobalVariable *
InstrLowerer::getOrCreateRegionBitmaps(InstrProfMCDCBitmapInstBase *Inc) {
  GlobalVariable *NamePtr = Inc->getName();
  auto &PD = ProfileDataMap[NamePtr];
  if (PD.RegionBitmaps)
    return PD.RegionBitmaps;

  // If RegionBitmaps doesn't already exist, create it by first setting up
  // the corresponding profile section.
  auto *BitmapPtr = setupProfileSection(Inc, IPSK_bitmap);
  PD.RegionBitmaps = BitmapPtr;
  PD.NumBitmapBytes = Inc->getNumBitmapBytes()->getZExtValue();
  return PD.RegionBitmaps;
}

GlobalVariable *
InstrLowerer::createRegionCounters(InstrProfCntrInstBase *Inc, StringRef Name,
                                   GlobalValue::LinkageTypes Linkage) {
  uint64_t NumCounters = Inc->getNumCounters()->getZExtValue();
  auto &Ctx = M.getContext();
  GlobalVariable *GV;
  if (isa<InstrProfCoverInst>(Inc)) {
    auto *CounterTy = Type::getInt8Ty(Ctx);
    auto *CounterArrTy = ArrayType::get(CounterTy, NumCounters);
    // TODO: `Constant::getAllOnesValue()` does not yet accept an array type.
    std::vector<Constant *> InitialValues(NumCounters,
                                          Constant::getAllOnesValue(CounterTy));
    GV = new GlobalVariable(M, CounterArrTy, false, Linkage,
                            ConstantArray::get(CounterArrTy, InitialValues),
                            Name);
    GV->setAlignment(Align(1));
  } else {
    auto *CounterTy = ArrayType::get(Type::getInt64Ty(Ctx), NumCounters);
    GV = new GlobalVariable(M, CounterTy, false, Linkage,
                            Constant::getNullValue(CounterTy), Name);
    GV->setAlignment(Align(8));
  }
  return GV;
}

GlobalVariable *
InstrLowerer::getOrCreateRegionCounters(InstrProfCntrInstBase *Inc) {
  GlobalVariable *NamePtr = Inc->getName();
  auto &PD = ProfileDataMap[NamePtr];
  if (PD.RegionCounters)
    return PD.RegionCounters;

  // If RegionCounters doesn't already exist, create it by first setting up
  // the corresponding profile section.
  auto *CounterPtr = setupProfileSection(Inc, IPSK_cnts);
  PD.RegionCounters = CounterPtr;

  if (DebugInfoCorrelate ||
      ProfileCorrelate == InstrProfCorrelator::DEBUG_INFO) {
    LLVMContext &Ctx = M.getContext();
    Function *Fn = Inc->getParent()->getParent();
    if (auto *SP = Fn->getSubprogram()) {
      DIBuilder DB(M, true, SP->getUnit());
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
    }

    // Mark the counter variable as used so that it isn't optimized out.
    CompilerUsedVars.push_back(PD.RegionCounters);
  }

  // Create the data variable (if it doesn't already exist).
  createDataVariable(Inc);

  return PD.RegionCounters;
}

void InstrLowerer::createDataVariable(InstrProfCntrInstBase *Inc) {
  // When debug information is correlated to profile data, a data variable
  // is not needed.
  if (DebugInfoCorrelate || ProfileCorrelate == InstrProfCorrelator::DEBUG_INFO)
    return;

  GlobalVariable *NamePtr = Inc->getName();
  auto &PD = ProfileDataMap[NamePtr];

  // Return if data variable was already created.
  if (PD.DataVar)
    return;

  LLVMContext &Ctx = M.getContext();

  Function *Fn = Inc->getParent()->getParent();
  GlobalValue::LinkageTypes Linkage = NamePtr->getLinkage();
  GlobalValue::VisibilityTypes Visibility = NamePtr->getVisibility();

  // Due to the limitation of binder as of 2021/09/28, the duplicate weak
  // symbols in the same csect won't be discarded. When there are duplicate weak
  // symbols, we can NOT guarantee that the relocations get resolved to the
  // intended weak symbol, so we can not ensure the correctness of the relative
  // CounterPtr, so we have to use private linkage for counter and data symbols.
  if (TT.isOSBinFormatXCOFF()) {
    Linkage = GlobalValue::PrivateLinkage;
    Visibility = GlobalValue::DefaultVisibility;
  }

  bool DataReferencedByCode = profDataReferencedByCode(M);
  bool NeedComdat = needsComdatForCounter(*Fn, M);
  bool Renamed;

  // The Data Variable section is anchored to profile counters.
  std::string CntsVarName =
      getVarName(Inc, getInstrProfCountersVarPrefix(), Renamed);
  std::string DataVarName =
      getVarName(Inc, getInstrProfDataVarPrefix(), Renamed);

  auto *Int8PtrTy = PointerType::getUnqual(Ctx);
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
        M, ValuesTy, false, Linkage, Constant::getNullValue(ValuesTy),
        getVarName(Inc, getInstrProfValuesVarPrefix(), Renamed));
    ValuesVar->setVisibility(Visibility);
    setGlobalVariableLargeSection(TT, *ValuesVar);
    ValuesVar->setSection(
        getInstrProfSectionName(IPSK_vals, TT.getObjectFormat()));
    ValuesVar->setAlignment(Align(8));
    maybeSetComdat(ValuesVar, Fn, CntsVarName);
    ValuesPtrExpr = ValuesVar;
  }

  uint64_t NumCounters = Inc->getNumCounters()->getZExtValue();
  auto *CounterPtr = PD.RegionCounters;

  uint64_t NumBitmapBytes = PD.NumBitmapBytes;

  // Create data variable.
  auto *IntPtrTy = M.getDataLayout().getIntPtrType(M.getContext());
  auto *Int16Ty = Type::getInt16Ty(Ctx);
  auto *Int16ArrayTy = ArrayType::get(Int16Ty, IPVK_Last + 1);
  Type *DataTypes[] = {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Init) LLVMType,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *DataTy = StructType::get(Ctx, ArrayRef(DataTypes));

  Constant *FunctionAddr = getFuncAddrForProfData(Fn);

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
      new GlobalVariable(M, DataTy, false, Linkage, nullptr, DataVarName);
  Constant *RelativeCounterPtr;
  GlobalVariable *BitmapPtr = PD.RegionBitmaps;
  Constant *RelativeBitmapPtr = ConstantInt::get(IntPtrTy, 0);
  InstrProfSectKind DataSectionKind;
  // With binary profile correlation, profile data is not loaded into memory.
  // profile data must reference profile counter with an absolute relocation.
  if (ProfileCorrelate == InstrProfCorrelator::BINARY) {
    DataSectionKind = IPSK_covdata;
    RelativeCounterPtr = ConstantExpr::getPtrToInt(CounterPtr, IntPtrTy);
    if (BitmapPtr != nullptr)
      RelativeBitmapPtr = ConstantExpr::getPtrToInt(BitmapPtr, IntPtrTy);
  } else {
    // Reference the counter variable with a label difference (link-time
    // constant).
    DataSectionKind = IPSK_data;
    RelativeCounterPtr =
        ConstantExpr::getSub(ConstantExpr::getPtrToInt(CounterPtr, IntPtrTy),
                             ConstantExpr::getPtrToInt(Data, IntPtrTy));
    if (BitmapPtr != nullptr)
      RelativeBitmapPtr =
          ConstantExpr::getSub(ConstantExpr::getPtrToInt(BitmapPtr, IntPtrTy),
                               ConstantExpr::getPtrToInt(Data, IntPtrTy));
  }

  Constant *DataVals[] = {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Init) Init,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  Data->setInitializer(ConstantStruct::get(DataTy, DataVals));

  Data->setVisibility(Visibility);
  Data->setSection(
      getInstrProfSectionName(DataSectionKind, TT.getObjectFormat()));
  Data->setAlignment(Align(INSTR_PROF_DATA_ALIGNMENT));
  maybeSetComdat(Data, Fn, CntsVarName);

  PD.DataVar = Data;

  // Mark the data variable as used so that it isn't stripped out.
  CompilerUsedVars.push_back(Data);
  // Now that the linkage set by the FE has been passed to the data and counter
  // variables, reset Name variable's linkage and visibility to private so that
  // it can be removed later by the compiler.
  NamePtr->setLinkage(GlobalValue::PrivateLinkage);
  // Collect the referenced names to be used by emitNameData.
  ReferencedNames.push_back(NamePtr);
}

void InstrLowerer::emitVNodes() {
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

  auto &Ctx = M.getContext();
  Type *VNodeTypes[] = {
#define INSTR_PROF_VALUE_NODE(Type, LLVMType, Name, Init) LLVMType,
#include "llvm/ProfileData/InstrProfData.inc"
  };
  auto *VNodeTy = StructType::get(Ctx, ArrayRef(VNodeTypes));

  ArrayType *VNodesTy = ArrayType::get(VNodeTy, NumCounters);
  auto *VNodesVar = new GlobalVariable(
      M, VNodesTy, false, GlobalValue::PrivateLinkage,
      Constant::getNullValue(VNodesTy), getInstrProfVNodesVarName());
  setGlobalVariableLargeSection(TT, *VNodesVar);
  VNodesVar->setSection(
      getInstrProfSectionName(IPSK_vnodes, TT.getObjectFormat()));
  VNodesVar->setAlignment(M.getDataLayout().getABITypeAlign(VNodesTy));
  // VNodesVar is used by runtime but not referenced via relocation by other
  // sections. Conservatively make it linker retained.
  UsedVars.push_back(VNodesVar);
}

void InstrLowerer::emitNameData() {
  std::string UncompressedData;

  if (ReferencedNames.empty())
    return;

  std::string CompressedNameStr;
  if (Error E = collectPGOFuncNameStrings(ReferencedNames, CompressedNameStr,
                                          DoInstrProfNameCompression)) {
    report_fatal_error(Twine(toString(std::move(E))), false);
  }

  auto &Ctx = M.getContext();
  auto *NamesVal =
      ConstantDataArray::getString(Ctx, StringRef(CompressedNameStr), false);
  NamesVar = new GlobalVariable(M, NamesVal->getType(), true,
                                GlobalValue::PrivateLinkage, NamesVal,
                                getInstrProfNamesVarName());
  NamesSize = CompressedNameStr.size();
  setGlobalVariableLargeSection(TT, *NamesVar);
  NamesVar->setSection(
      ProfileCorrelate == InstrProfCorrelator::BINARY
          ? getInstrProfSectionName(IPSK_covname, TT.getObjectFormat())
          : getInstrProfSectionName(IPSK_name, TT.getObjectFormat()));
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

void InstrLowerer::emitRegistration() {
  if (!needsRuntimeRegistrationOfSectionRange(TT))
    return;

  // Construct the function.
  auto *VoidTy = Type::getVoidTy(M.getContext());
  auto *VoidPtrTy = PointerType::getUnqual(M.getContext());
  auto *Int64Ty = Type::getInt64Ty(M.getContext());
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

  IRBuilder<> IRB(BasicBlock::Create(M.getContext(), "", RegisterF));
  for (Value *Data : CompilerUsedVars)
    if (!isa<Function>(Data))
      IRB.CreateCall(RuntimeRegisterF, Data);
  for (Value *Data : UsedVars)
    if (Data != NamesVar && !isa<Function>(Data))
      IRB.CreateCall(RuntimeRegisterF, Data);

  if (NamesVar) {
    Type *ParamTypes[] = {VoidPtrTy, Int64Ty};
    auto *NamesRegisterTy =
        FunctionType::get(VoidTy, ArrayRef(ParamTypes), false);
    auto *NamesRegisterF =
        Function::Create(NamesRegisterTy, GlobalVariable::ExternalLinkage,
                         getInstrProfNamesRegFuncName(), M);
    IRB.CreateCall(NamesRegisterF, {NamesVar, IRB.getInt64(NamesSize)});
  }

  IRB.CreateRetVoid();
}

bool InstrLowerer::emitRuntimeHook() {
  // We expect the linker to be invoked with -u<hook_var> flag for Linux
  // in which case there is no need to emit the external variable.
  if (TT.isOSLinux() || TT.isOSAIX())
    return false;

  // If the module's provided its own runtime, we don't need to do anything.
  if (M.getGlobalVariable(getInstrProfRuntimeHookVarName()))
    return false;

  // Declare an external variable that will pull in the runtime initialization.
  auto *Int32Ty = Type::getInt32Ty(M.getContext());
  auto *Var =
      new GlobalVariable(M, Int32Ty, false, GlobalValue::ExternalLinkage,
                         nullptr, getInstrProfRuntimeHookVarName());
  Var->setVisibility(GlobalValue::HiddenVisibility);

  if (TT.isOSBinFormatELF() && !TT.isPS()) {
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
      User->setComdat(M.getOrInsertComdat(User->getName()));

    IRBuilder<> IRB(BasicBlock::Create(M.getContext(), "", User));
    auto *Load = IRB.CreateLoad(Int32Ty, Var);
    IRB.CreateRet(Load);

    // Mark the function as used so that it isn't stripped out.
    CompilerUsedVars.push_back(User);
  }
  return true;
}

void InstrLowerer::emitUses() {
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
      (TT.isOSBinFormatCOFF() && !profDataReferencedByCode(M)))
    appendToCompilerUsed(M, CompilerUsedVars);
  else
    appendToUsed(M, CompilerUsedVars);

  // We do not add proper references from used metadata sections to NamesVar and
  // VNodesVar, so we have to be conservative and place them in llvm.used
  // regardless of the target,
  appendToUsed(M, UsedVars);
}

void InstrLowerer::emitInitialization() {
  // Create ProfileFileName variable. Don't don't this for the
  // context-sensitive instrumentation lowering: This lowering is after
  // LTO/ThinLTO linking. Pass PGOInstrumentationGenCreateVar should
  // have already create the variable before LTO/ThinLTO linking.
  if (!IsCS)
    createProfileFileNameVar(M, Options.InstrProfileOutput);
  Function *RegisterF = M.getFunction(getInstrProfRegFuncsName());
  if (!RegisterF)
    return;

  // Create the initialization function.
  auto *VoidTy = Type::getVoidTy(M.getContext());
  auto *F = Function::Create(FunctionType::get(VoidTy, false),
                             GlobalValue::InternalLinkage,
                             getInstrProfInitFuncName(), M);
  F->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  F->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone)
    F->addFnAttr(Attribute::NoRedZone);

  // Add the basic block and the necessary calls.
  IRBuilder<> IRB(BasicBlock::Create(M.getContext(), "", F));
  IRB.CreateCall(RegisterF, {});
  IRB.CreateRetVoid();

  appendToGlobalCtors(M, F, 0);
}
