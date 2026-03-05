//===-- AMDGPURegPressureEstimator.cpp - AMDGPU Reg Pressure -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Estimates VGPR register pressure at IR level for AMDGPURegPressureGuard.
/// Uses RPO dataflow analysis to track live values through the function.
/// Results are relative only - not comparable to hardware register counts.
///
//===----------------------------------------------------------------------===//

#include "AMDGPURegPressureEstimator.h"
#include "AMDGPU.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-reg-pressure-estimator"

namespace {
// Returns VGPR cost in half-registers (16-bit units).
// Returns 0 for SGPRs, constants, uniform values, and i1 types.
static unsigned getVgprCost(Value *V, const DataLayout &DL,
                            const UniformityInfo &UA, bool UseRealTrue16) {
  if (!V)
    return 0;

  Type *Ty = V->getType();
  if (Ty->isVoidTy() || Ty->isTokenTy() || Ty->isMetadataTy() ||
      !Ty->isSized() || Ty->isIntegerTy(1))
    return 0;

  if (UA.isUniform(V) || isa<CmpInst>(V))
    return 0;

  if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
    unsigned AS = PtrTy->getAddressSpace();
    switch (AS) {
    case AMDGPUAS::BUFFER_FAT_POINTER:
      return 2; // offset
    case AMDGPUAS::BUFFER_RESOURCE:
      return 0;
    case AMDGPUAS::BUFFER_STRIDED_POINTER:
      return 4; // offset + index
    default:
      unsigned BitWidth = DL.getPointerSizeInBits(AS);
      return ((BitWidth + 31) / 32) * 2;
    }
  }

  unsigned BitWidth = DL.getTypeStoreSizeInBits(Ty).getFixedValue();
  if (Ty->isIntegerTy())
    return ((BitWidth + 31) / 32) * 2;

  if (UseRealTrue16)
    return (BitWidth + 15) / 16;
  return ((BitWidth + 31) / 32) * 2;
}

// Caches block-to-block reachability queries to avoid redundant BFS traversals.
// Uses RPO indices to quickly reject backward reachability in acyclic regions.
class ReachabilityCache {
  DenseMap<std::pair<BasicBlock *, BasicBlock *>, bool> BBCache;
  PostDominatorTree *PDT;

  DenseMap<BasicBlock *, unsigned> RPOIndex;
  bool HasBackEdges = false;

public:
  DominatorTree &DT;

  ReachabilityCache(DominatorTree &DT, PostDominatorTree *PDT)
      : PDT(PDT), DT(DT) {}

  void initRPO(ReversePostOrderTraversal<Function *> &RPOT) {
    unsigned Idx = 0;
    for (auto *BB : RPOT)
      RPOIndex[BB] = Idx++;

    for (auto *BB : RPOT) {
      unsigned FromIdx = RPOIndex[BB];
      for (BasicBlock *Succ : successors(BB)) {
        if (RPOIndex[Succ] <= FromIdx) {
          HasBackEdges = true;
          return;
        }
      }
    }
  }

  bool isReachable(Instruction *FromInst, Instruction *ToInst) {
    BasicBlock *FromBB = FromInst->getParent();
    BasicBlock *ToBB = ToInst->getParent();

    if (FromBB == ToBB)
      return FromInst->comesBefore(ToInst);

    auto Key = std::make_pair(FromBB, ToBB);
    auto It = BBCache.find(Key);
    if (It != BBCache.end())
      return It->second;

    auto CacheAndReturn = [&](bool Result) {
      BBCache[Key] = Result;
      return Result;
    };

    if (DT.dominates(ToBB, FromBB))
      return CacheAndReturn(false);

    if (PDT && PDT->dominates(FromBB, ToBB))
      return CacheAndReturn(false);

    if (!HasBackEdges && !RPOIndex.empty()) {
      auto FromIt = RPOIndex.find(FromBB);
      auto ToIt = RPOIndex.find(ToBB);
      if (FromIt != RPOIndex.end() && ToIt != RPOIndex.end()) {
        if (FromIt->second > ToIt->second)
          return CacheAndReturn(false);
      }
    }

    return CacheAndReturn(computeReachability(FromBB, ToBB));
  }

private:
  bool computeReachability(BasicBlock *FromBB, BasicBlock *ToBB) {
    SmallPtrSet<BasicBlock *, 32> Visited;
    SmallVector<BasicBlock *, 16> Worklist;

    for (BasicBlock *Succ : successors(FromBB)) {
      if (Succ == ToBB)
        return true;
      Worklist.push_back(Succ);
      Visited.insert(Succ);
    }

    Visited.insert(FromBB);

    while (!Worklist.empty()) {
      BasicBlock *BB = Worklist.pop_back_val();

      for (BasicBlock *Succ : successors(BB)) {
        if (Succ == ToBB)
          return true;

        if (Visited.count(Succ))
          continue;

        if (DT.dominates(Succ, FromBB))
          continue;

        Visited.insert(Succ);
        Worklist.push_back(Succ);
      }
    }

    return false;
  }
};

// Checks if a value becomes dead after a specific instruction.
// Returns true if V has no uses reachable from AfterInst, meaning V's
// live range ends at AfterInst and can be removed from the pressure tracking.
static bool isValueDead(Value *V, Instruction *AfterInst,
                        ReachabilityCache &Cache) {
  for (User *U : V->users()) {
    Instruction *UseInst = dyn_cast<Instruction>(U);
    if (!UseInst)
      continue;

    if (UseInst == AfterInst)
      continue;

    if (Cache.DT.dominates(UseInst, AfterInst))
      continue;

    if (Cache.isReachable(AfterInst, UseInst))
      return false;
  }

  return true;
}

// Estimates VGPR register pressure using forward dataflow analysis in RPO.
// Tracks live value ranges to compute pressure at each program point.
class AMDGPURegPressureEstimator {
private:
  Function &F;
  DominatorTree &DT;
  PostDominatorTree *PDT;
  const UniformityInfo &UA;
  const DataLayout &DL;

  DenseSet<Value *> GlobalDeadSet;
  ReachabilityCache ReachCache;
  unsigned MaxPressureHalfRegs = 0;
  bool UseRealTrue16;

public:
  AMDGPURegPressureEstimator(Function &F, DominatorTree &DT,
                             PostDominatorTree *PDT, const UniformityInfo &UA)
      : F(F), DT(DT), PDT(PDT), UA(UA), DL(F.getParent()->getDataLayout()),
        ReachCache(DT, PDT) {
    // Check if real-true16 feature is enabled for this function.
    Attribute FSAttr = F.getFnAttribute("target-features");
    UseRealTrue16 = FSAttr.isValid() &&
                    FSAttr.getValueAsString().contains("+real-true16");
  }

  unsigned getMaxVGPRs() const { return (MaxPressureHalfRegs + 1) / 2; }

  void analyze() {
    LLVM_DEBUG(dbgs() << "Analyzing function: " << F.getName() << "\n");

    // Main algorithm: Forward dataflow analysis in reverse post-order (RPO)
    //
    // RPO traversal ensures that:
    // 1. Each block is visited after all its predecessors (except back edges)
    // 2. Loop headers are visited before loop bodies
    // 3. This allows accurate propagation of live value information
    //
    // For each basic block, we:
    // 1. Merge live-in states from all predecessors
    // 2. Process instructions sequentially, tracking:
    //    - New values becoming live (instruction results)
    //    - Values becoming dead (last use detected)
    //    - Special handling for insert/extract operations
    // 3. Record live-out state for use by successors
    // 4. Track maximum pressure seen at any program point
    DenseMap<BasicBlock *, DenseMap<Value *, unsigned>> BlockExitStates;

    ReversePostOrderTraversal<Function *> RPOT(&F);
    ReachCache.initRPO(RPOT);

    BasicBlock *EntryBB = &F.getEntryBlock();
    DenseMap<Value *, unsigned> EntryLiveMap;
    for (Argument &Arg : F.args()) {
      if (Arg.use_empty())
        continue;
      unsigned Cost = getVgprCost(&Arg, DL, UA, UseRealTrue16);
      if (Cost > 0)
        EntryLiveMap[&Arg] = Cost;
    }

    for (auto It = RPOT.begin(), E = RPOT.end(); It != E; ++It) {
      BasicBlock *BB = *It;
      DenseMap<Value *, unsigned> BlockEntryLiveMap;

      if (BB == EntryBB)
        BlockEntryLiveMap = EntryLiveMap;
      else {
        for (BasicBlock *Pred : predecessors(BB)) {
          auto PredIt = BlockExitStates.find(Pred);
          if (PredIt == BlockExitStates.end())
            continue;

          const DenseMap<Value *, unsigned> &PredExitMap = PredIt->second;
          for (auto &[V, Cost] : PredExitMap) {
            if (GlobalDeadSet.count(V))
              continue;

            BlockEntryLiveMap[V] = Cost;
          }
        }
      }

      DenseMap<Value *, unsigned> ExitLiveMap =
          processBlock(*BB, BlockEntryLiveMap);

      BlockExitStates[BB] = ExitLiveMap;
    }

    LLVM_DEBUG(dbgs() << "  Max pressure: " << (MaxPressureHalfRegs / 2)
                      << " VGPRs\n");
  }

private:
  static std::string getBBName(const BasicBlock *BB) {
    if (!BB->getName().empty())
      return BB->getName().str();

    std::string Name;
    raw_string_ostream OS(Name);
    BB->printAsOperand(OS, false);
    if (!Name.empty() && Name[0] == '%')
      Name.erase(Name.begin());
    return Name;
  }

  DenseMap<Value *, unsigned>
  processBlock(BasicBlock &BB, DenseMap<Value *, unsigned> InitialLiveMap) {
    DenseMap<Value *, unsigned> CurrentLiveMap = InitialLiveMap;

    unsigned CurrentPressure = computePressure(CurrentLiveMap);

    if (CurrentPressure > MaxPressureHalfRegs)
      MaxPressureHalfRegs = CurrentPressure;

    for (Instruction &I : BB) {
      if (I.isDebugOrPseudoInst())
        continue;

      // Process instruction result: determine if it creates a new live VGPR value
      //
      // Three cases with different pressure impacts:
      //
      // Case 1: Insert operations (insertelement, insertvalue)
      //   - Conservative approach: assume both aggregate and inserted value are live
      //   - Pressure increase = aggregate_cost + inserted_value_cost
      //   - Rationale: Without dataflow, we can't track which vector lanes are live
      //
      // Case 2: Extract operations (extractelement, extractvalue)
      //   - Extract creates a new live value (extract_cost)
      //   - Source aggregate cost can be reduced by extract_cost
      //   - Handles partial liveness of vectors/aggregates
      //   - Only applies if source is a tracked VGPR value
      //
      // Case 3: Other operations
      //   - Result becomes live only if it has at least one live VGPR operand
      //   - This prevents counting operations on uniform/constant values
      if (!I.getType()->isVoidTy() && !I.use_empty()) {
        if (isa<InsertElementInst>(&I) || isa<InsertValueInst>(&I)) {
          Value *Aggregate = nullptr;
          Value *InsertedVal = nullptr;

          if (auto *IEI = dyn_cast<InsertElementInst>(&I)) {
            Aggregate = IEI->getOperand(0);
            InsertedVal = IEI->getOperand(1);
          } else if (auto *IVI = dyn_cast<InsertValueInst>(&I)) {
            Aggregate = IVI->getAggregateOperand();
            InsertedVal = IVI->getInsertedValueOperand();
          }

          unsigned AggCost = CurrentLiveMap.lookup(Aggregate);
          unsigned InsertedCost = CurrentLiveMap.lookup(InsertedVal);
          unsigned NewCost = AggCost + InsertedCost;

          if (NewCost > 0) {
            CurrentLiveMap[&I] = NewCost;
            CurrentPressure += NewCost;
          }
        } else if (isa<ExtractValueInst>(&I) || isa<ExtractElementInst>(&I)) {
          Value *Source = nullptr;
          if (auto *EVI = dyn_cast<ExtractValueInst>(&I))
            Source = EVI->getAggregateOperand();
          else if (auto *EEI = dyn_cast<ExtractElementInst>(&I))
            Source = EEI->getVectorOperand();

          bool IsSourceVGPR = Source && CurrentLiveMap.count(Source);

          unsigned ExtractCost = getVgprCost(&I, DL, UA, UseRealTrue16);

          if (ExtractCost > 0 && IsSourceVGPR) {
            CurrentLiveMap[&I] = ExtractCost;
            CurrentPressure += ExtractCost;

            auto SourceIt = CurrentLiveMap.find(Source);
            if (SourceIt != CurrentLiveMap.end()) {
              unsigned OldCost = SourceIt->second;
              if (OldCost >= ExtractCost) {
                SourceIt->second -= ExtractCost;
                CurrentPressure -= ExtractCost;

                if (SourceIt->second == 0)
                  CurrentLiveMap.erase(SourceIt);
              }
            }
          }
        } else {
          bool HasLiveVgprOperand = false;
          for (Use &Op : I.operands()) {
            Value *V = Op.get();
            if (isa<Constant>(V) || isa<BasicBlock>(V))
              continue;
            if (CurrentLiveMap.count(V)) {
              HasLiveVgprOperand = true;
              break;
            }
          }

          if (HasLiveVgprOperand) {
            unsigned Cost = getVgprCost(&I, DL, UA, UseRealTrue16);

            if (Cost > 0) {
              CurrentLiveMap[&I] = Cost;
              CurrentPressure += Cost;
            }
          }
        }
      }

      // Kill dead values: check if any operand's last use is at this instruction
      for (Use &Op : I.operands()) {
        Value *V = Op.get();
        if (isa<Constant>(V) || isa<BasicBlock>(V))
          continue;

        auto It = CurrentLiveMap.find(V);
        if (It != CurrentLiveMap.end()) {
          bool IsDead = isValueDead(V, &I, ReachCache);

          if (IsDead) {
            unsigned Cost = It->second;

            CurrentLiveMap.erase(It);
            CurrentPressure -= Cost;

            GlobalDeadSet.insert(V);
          }
        }
      }

      if (CurrentPressure > MaxPressureHalfRegs)
        MaxPressureHalfRegs = CurrentPressure;
    }

    return CurrentLiveMap;
  }

  unsigned computePressure(const DenseMap<Value *, unsigned> &LiveMap) {
    unsigned Total = 0;
    for (auto &[V, Cost] : LiveMap)
      Total += Cost;
    return Total;
  }
};

static unsigned computeMaxVGPRPressure(Function &F, DominatorTree &DT,
                                       PostDominatorTree *PDT,
                                       const UniformityInfo &UA) {
  AMDGPURegPressureEstimator Estimator(F, DT, PDT, UA);
  Estimator.analyze();
  return Estimator.getMaxVGPRs();
}

} // end anonymous namespace

namespace llvm {

AnalysisKey AMDGPURegPressureEstimatorAnalysis::Key;

AMDGPURegPressureEstimatorAnalysis::Result
AMDGPURegPressureEstimatorAnalysis::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  auto &UA = AM.getResult<UniformityInfoAnalysis>(F);

  unsigned MaxVGPRs = computeMaxVGPRPressure(F, DT, &PDT, UA);
  return AMDGPURegPressureEstimatorResult(MaxVGPRs);
}

PreservedAnalyses
AMDGPURegPressureEstimatorPrinterPass::run(Function &F,
                                           FunctionAnalysisManager &AM) {
  auto Result = AM.getResult<AMDGPURegPressureEstimatorAnalysis>(F);
  OS << "AMDGPU Register Pressure for function '" << F.getName()
     << "': " << Result.MaxVGPRs << " VGPRs (IR-level estimate)\n";
  return PreservedAnalyses::all();
}

char AMDGPURegPressureEstimatorWrapperPass::ID = 0;

AMDGPURegPressureEstimatorWrapperPass::AMDGPURegPressureEstimatorWrapperPass()
    : FunctionPass(ID) {
  initializeAMDGPURegPressureEstimatorWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

bool AMDGPURegPressureEstimatorWrapperPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto *PDTPass = getAnalysisIfAvailable<PostDominatorTreeWrapperPass>();
  PostDominatorTree *PDT = PDTPass ? &PDTPass->getPostDomTree() : nullptr;
  auto &UA = getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();

  MaxVGPRs = computeMaxVGPRPressure(F, DT, PDT, UA);
  return false;
}

void AMDGPURegPressureEstimatorWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<UniformityInfoWrapperPass>();
  AU.addUsedIfAvailable<PostDominatorTreeWrapperPass>();
}

void AMDGPURegPressureEstimatorWrapperPass::print(raw_ostream &OS,
                                                  const Module *) const {
  OS << "AMDGPU Register Pressure: " << MaxVGPRs
     << " VGPRs (IR-level estimate)\n";
}

INITIALIZE_PASS_BEGIN(AMDGPURegPressureEstimatorWrapperPass,
                      "amdgpu-reg-pressure-estimator",
                      "AMDGPU Register Pressure Estimator", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPURegPressureEstimatorWrapperPass,
                    "amdgpu-reg-pressure-estimator",
                    "AMDGPU Register Pressure Estimator", false, true)

} // end namespace llvm
