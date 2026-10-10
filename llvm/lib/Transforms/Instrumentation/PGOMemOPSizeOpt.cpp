//===-- PGOMemOPSizeOpt.cpp - Optimizations based on value profiling ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the transformation that optimizes memory intrinsics
// such as memcpy using the size value profile. When memory intrinsic size
// value profile metadata is available, a single memory intrinsic is expanded
// to a sequence of guarded specialized versions that are called with the
// hottest size(s), for later expansion into more optimal inline sequences.
//
//===----------------------------------------------------------------------===//

#include "InstrumentationOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/ProfileData/InstrProf.h"
#define INSTR_PROF_VALUE_PROF_MEMOP_API
#include "llvm/ProfileData/InstrProfData.inc"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <cassert>
#include <cstdint>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pgo-memop-opt"

STATISTIC(NumOfPGOMemOPOpt, "Number of memop intrinsics optimized.");
STATISTIC(NumOfPGOMemOPAnnotate, "Number of memop intrinsics annotated.");

namespace llvm {

} // end namespace llvm

namespace {

static const char *getMIName(const MemIntrinsic *MI) {
  switch (MI->getIntrinsicID()) {
  case Intrinsic::memcpy:
    return "memcpy";
  case Intrinsic::memmove:
    return "memmove";
  case Intrinsic::memset:
    return "memset";
  default:
    return "unknown";
  }
}

// A class that abstracts a memop (memcpy, memmove, memset, memcmp and bcmp).
struct MemOp {
  Instruction *I;
  MemOp(MemIntrinsic *MI) : I(MI) {}
  MemOp(CallInst *CI) : I(CI) {}
  MemIntrinsic *asMI() { return dyn_cast<MemIntrinsic>(I); }
  CallInst *asCI() { return cast<CallInst>(I); }
  MemOp clone() {
    if (auto MI = asMI())
      return MemOp(cast<MemIntrinsic>(MI->clone()));
    return MemOp(cast<CallInst>(asCI()->clone()));
  }
  Value *getLength() {
    if (auto MI = asMI())
      return MI->getLength();
    return asCI()->getArgOperand(2);
  }
  void setLength(Value *Length) {
    if (auto MI = asMI())
      return MI->setLength(Length);
    asCI()->setArgOperand(2, Length);
  }
  StringRef getFuncName() {
    if (auto MI = asMI())
      return MI->getCalledFunction()->getName();
    return asCI()->getCalledFunction()->getName();
  }
  bool isMemmove() {
    if (auto MI = asMI())
      if (MI->getIntrinsicID() == Intrinsic::memmove)
        return true;
    return false;
  }
  bool isMemcmp(TargetLibraryInfo &TLI) {
    return asMI() == nullptr && TLI.getLibFunc(*asCI()) == LibFunc_memcmp;
  }
  bool isBcmp(TargetLibraryInfo &TLI) {
    return asMI() == nullptr && TLI.getLibFunc(*asCI()) == LibFunc_bcmp;
  }
  const char *getName(TargetLibraryInfo &TLI) {
    if (auto MI = asMI())
      return getMIName(MI);
    LibFunc Func = TLI.getLibFunc(*asCI());
    if (Func == LibFunc_memcmp)
      return "memcmp";
    if (Func == LibFunc_bcmp)
      return "bcmp";
    llvm_unreachable("Must be MemIntrinsic or memcmp/bcmp CallInst");
    return nullptr;
  }
};

class MemOPSizeOpt : public InstVisitor<MemOPSizeOpt> {
public:
  MemOPSizeOpt(const InstrumentationOptions &Opts, Function &Func,
               BlockFrequencyInfo &BFI, OptimizationRemarkEmitter &ORE,
               DominatorTree *DT, TargetLibraryInfo &TLI)
      : Opts(Opts), Func(Func), BFI(BFI), ORE(ORE), DT(DT), TLI(TLI),
        Changed(false) {}
  bool isChanged() const { return Changed; }
  void perform() {
    WorkList.clear();
    visit(Func);

    for (auto &MO : WorkList) {
      ++NumOfPGOMemOPAnnotate;
      if (perform(MO)) {
        Changed = true;
        ++NumOfPGOMemOPOpt;
        LLVM_DEBUG(dbgs() << "MemOP call: " << MO.getFuncName()
                          << "is Transformed.\n");
      }
    }
  }

  void visitMemIntrinsic(MemIntrinsic &MI) {
    Value *Length = MI.getLength();
    // Not perform on constant length calls.
    if (isa<ConstantInt>(Length))
      return;
    WorkList.push_back(MemOp(&MI));
  }

  void visitCallInst(CallInst &CI) {
    LibFunc Func = TLI.getLibFunc(CI);
    if ((Func == LibFunc_memcmp || Func == LibFunc_bcmp) &&
        !isa<ConstantInt>(CI.getArgOperand(2))) {
      WorkList.push_back(MemOp(&CI));
    }
  }

private:
  const InstrumentationOptions &Opts;
  Function &Func;
  BlockFrequencyInfo &BFI;
  OptimizationRemarkEmitter &ORE;
  DominatorTree *DT;
  TargetLibraryInfo &TLI;
  bool Changed;
  std::vector<MemOp> WorkList;
  bool perform(MemOp MO);
};

static bool isProfitable(const InstrumentationOptions &Opts, uint64_t Count,
                         uint64_t TotalCount) {
  assert(Count <= TotalCount);
  if (Count < Opts.pgo_memop_count_threshold)
    return false;
  if (Count < TotalCount * Opts.pgo_memop_percent_threshold / 100)
    return false;
  return true;
}

static inline uint64_t getScaledCount(const InstrumentationOptions &Opts,
                                      uint64_t Count, uint64_t Num,
                                      uint64_t Denom) {
  if (!Opts.pgo_memop_scale_count)
    return Count;
  bool Overflowed;
  uint64_t ScaleCount = SaturatingMultiply(Count, Num, &Overflowed);
  return ScaleCount / Denom;
}

bool MemOPSizeOpt::perform(MemOp MO) {
  assert(MO.I);
  if (MO.isMemmove())
    return false;
  if (!Opts.pgo_memop_optimize_memcmp_bcmp &&
      (MO.isMemcmp(TLI) || MO.isBcmp(TLI)))
    return false;

  uint32_t MaxNumVals = INSTR_PROF_NUM_BUCKETS;
  uint64_t TotalCount;
  auto VDs =
      getValueProfDataFromInst(*MO.I, IPVK_MemOPSize, MaxNumVals, TotalCount);
  if (VDs.empty())
    return false;

  uint64_t ActualCount = TotalCount;
  uint64_t SavedTotalCount = TotalCount;
  if (Opts.pgo_memop_scale_count) {
    auto BBEdgeCount = BFI.getBlockProfileCount(MO.I->getParent());
    if (!BBEdgeCount)
      return false;
    ActualCount = *BBEdgeCount;
  }

  LLVM_DEBUG(dbgs() << "Read one memory intrinsic profile with count "
                    << ActualCount << "\n");
  LLVM_DEBUG(
      for (auto &VD
           : VDs) { dbgs() << "  (" << VD.Value << "," << VD.Count << ")\n"; });

  if (ActualCount < Opts.pgo_memop_count_threshold)
    return false;
  // Skip if the total value profiled count is 0, in which case we can't
  // scale up the counts properly (and there is no profitable transformation).
  if (TotalCount == 0)
    return false;

  TotalCount = ActualCount;
  if (Opts.pgo_memop_scale_count)
    LLVM_DEBUG(dbgs() << "Scale counts: numerator = " << ActualCount
                      << " denominator = " << SavedTotalCount << "\n");

  // Keeping track of the count of the default case:
  uint64_t RemainCount = TotalCount;
  uint64_t SavedRemainCount = SavedTotalCount;
  SmallVector<uint64_t, 16> SizeIds;
  SmallVector<uint64_t, 16> CaseCounts;
  uint64_t MaxCount = 0;
  unsigned Version = 0;
  // Default case is in the front -- save the slot here.
  CaseCounts.push_back(0);
  SmallVector<InstrProfValueData, 24> RemainingVDs;
  for (auto I = VDs.begin(), E = VDs.end(); I != E; ++I) {
    auto &VD = *I;
    int64_t V = VD.Value;
    uint64_t C = VD.Count;
    if (Opts.pgo_memop_scale_count)
      C = getScaledCount(Opts, C, ActualCount, SavedTotalCount);

    if (!InstrProfIsSingleValRange(V) ||
        V > Opts.memop_value_prof_max_opt_size) {
      RemainingVDs.push_back(VD);
      continue;
    }

    // ValueCounts are sorted on the count. Break at the first un-profitable
    // value.
    if (!isProfitable(Opts, C, RemainCount)) {
      RemainingVDs.insert(RemainingVDs.end(), I, E);
      break;
    }

    SizeIds.push_back(V);
    CaseCounts.push_back(C);
    if (C > MaxCount)
      MaxCount = C;

    assert(RemainCount >= C);
    RemainCount -= C;
    assert(SavedRemainCount >= VD.Count);
    SavedRemainCount -= VD.Count;

    if (++Version >= Opts.pgo_memop_max_version &&
        Opts.pgo_memop_max_version != 0) {
      RemainingVDs.insert(RemainingVDs.end(), I + 1, E);
      break;
    }
  }

  if (Version == 0)
    return false;

  CaseCounts[0] = RemainCount;
  if (RemainCount > MaxCount)
    MaxCount = RemainCount;

  uint64_t SumForOpt = TotalCount - RemainCount;

  LLVM_DEBUG(dbgs() << "Optimize one memory intrinsic call to " << Version
                    << " Versions (covering " << SumForOpt << " out of "
                    << TotalCount << ")\n");

  // mem_op(..., size)
  // ==>
  // switch (size) {
  //   case s1:
  //      mem_op(..., s1);
  //      goto merge_bb;
  //   case s2:
  //      mem_op(..., s2);
  //      goto merge_bb;
  //   ...
  //   default:
  //      mem_op(..., size);
  //      goto merge_bb;
  // }
  // merge_bb:

  BasicBlock *BB = MO.I->getParent();
  LLVM_DEBUG(dbgs() << "\n\n== Basic Block Before ==\n");
  LLVM_DEBUG(dbgs() << *BB << "\n");
  auto OrigBBFreq = BFI.getBlockFreq(BB);

  BasicBlock *DefaultBB = SplitBlock(BB, MO.I, DT);
  BasicBlock::iterator It(*MO.I);
  ++It;
  assert(It != DefaultBB->end());
  BasicBlock *MergeBB = SplitBlock(DefaultBB, &(*It), DT);
  MergeBB->setName("MemOP.Merge");
  BFI.setBlockFreq(MergeBB, OrigBBFreq);
  DefaultBB->setName("MemOP.Default");

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  auto &Ctx = Func.getContext();
  IRBuilder<> IRB(BB);
  BB->getTerminator()->eraseFromParent();
  Value *SizeVar = MO.getLength();
  SwitchInst *SI = IRB.CreateSwitch(SizeVar, DefaultBB, SizeIds.size());
  Type *MemOpTy = MO.I->getType();
  PHINode *PHI = nullptr;
  if (!MemOpTy->isVoidTy()) {
    // Insert a phi for the return values at the merge block.
    IRBuilder<> IRBM(MergeBB->getFirstNonPHIIt());
    PHI = IRBM.CreatePHI(MemOpTy, SizeIds.size() + 1, "MemOP.RVMerge");
    MO.I->replaceAllUsesWith(PHI);
    PHI->addIncoming(MO.I, DefaultBB);
  }

  // Clear the value profile data.
  MO.I->setMetadata(LLVMContext::MD_prof, nullptr);
  // If all promoted, we don't need the MD.prof metadata.
  if (SavedRemainCount > 0 || Version != VDs.size()) {
    // Otherwise we need update with the un-promoted records back.
    annotateValueSite(*Func.getParent(), *MO.I, RemainingVDs, SavedRemainCount,
                      IPVK_MemOPSize, VDs.size());
  }

  LLVM_DEBUG(dbgs() << "\n\n== Basic Block After==\n");

  std::vector<DominatorTree::UpdateType> Updates;
  if (DT)
    Updates.reserve(2 * SizeIds.size());

  for (uint64_t SizeId : SizeIds) {
    BasicBlock *CaseBB = BasicBlock::Create(
        Ctx, Twine("MemOP.Case.") + Twine(SizeId), &Func, DefaultBB);
    MemOp NewMO = MO.clone();
    // Fix the argument.
    auto *SizeType = dyn_cast<IntegerType>(NewMO.getLength()->getType());
    assert(SizeType && "Expected integer type size argument.");
    ConstantInt *CaseSizeId = ConstantInt::get(SizeType, SizeId);
    NewMO.setLength(CaseSizeId);
    NewMO.I->insertInto(CaseBB, CaseBB->end());
    IRBuilder<> IRBCase(CaseBB);
    IRBCase.CreateBr(MergeBB);
    SI->addCase(CaseSizeId, CaseBB);
    if (!MemOpTy->isVoidTy())
      PHI->addIncoming(NewMO.I, CaseBB);
    if (DT) {
      Updates.push_back({DominatorTree::Insert, CaseBB, MergeBB});
      Updates.push_back({DominatorTree::Insert, BB, CaseBB});
    }
    LLVM_DEBUG(dbgs() << *CaseBB << "\n");
  }
  DTU.applyUpdates(Updates);
  Updates.clear();

  if (MaxCount)
    setProfMetadata(SI, CaseCounts, MaxCount);

  LLVM_DEBUG(dbgs() << *BB << "\n");
  LLVM_DEBUG(dbgs() << *DefaultBB << "\n");
  LLVM_DEBUG(dbgs() << *MergeBB << "\n");

  ORE.emit([&]() {
    using namespace ore;
    return OptimizationRemark(DEBUG_TYPE, "memopt-opt", MO.I)
           << "optimized " << NV("Memop", MO.getName(TLI)) << " with count "
           << NV("Count", SumForOpt) << " out of " << NV("Total", TotalCount)
           << " for " << NV("Versions", Version) << " versions";
  });

  return true;
}
} // namespace

static bool pgoMemOPSizeOptImpl(const InstrumentationOptions &Opts, Function &F,
                                BlockFrequencyInfo &BFI,
                                OptimizationRemarkEmitter &ORE,
                                DominatorTree *DT, TargetLibraryInfo &TLI) {
  if (Opts.disable_memop_opt)
    return false;

  if (F.hasOptSize())
    return false;
  MemOPSizeOpt MemOPSizeOpt(Opts, F, BFI, ORE, DT, TLI);
  MemOPSizeOpt.perform();
  return MemOPSizeOpt.isChanged();
}

PreservedAnalyses PGOMemOPSizeOpt::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  auto &BFI = FAM.getResult<BlockFrequencyAnalysis>(F);
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  auto *DT = FAM.getCachedResult<DominatorTreeAnalysis>(F);
  auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
  bool Changed =
      pgoMemOPSizeOptImpl(InstrumentationOptions::Global, F, BFI, ORE, DT, TLI);
  if (!Changed)
    return PreservedAnalyses::all();
  auto PA = PreservedAnalyses();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
