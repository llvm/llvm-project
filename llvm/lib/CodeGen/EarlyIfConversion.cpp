//===-- EarlyIfConversion.cpp - If-conversion on SSA form machine code ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Early if-conversion is for out-of-order CPUs that don't have a lot of
// predicable instructions. The goal is to eliminate conditional branches that
// may mispredict.
//
// Instructions from both sides of the branch are executed specutatively, and a
// cmov instruction selects the result.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SSAIfConv.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "early-ifcvt"

namespace {
struct EarlyIfConverter : MachineFunctionPass {
  static char ID;
  EarlyIfConverter() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return "Early If-Conversion"; }
};

char EarlyIfConverter::ID = 0;

void EarlyIfConverter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  AU.addRequired<MachineTraceMetrics>();
  AU.addPreserved<MachineTraceMetrics>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

// Adjust cycles with downward saturation.
unsigned adjCycles(unsigned Cyc, int Delta) {
  if (Delta < 0 && Cyc + Delta > Cyc)
    return 0;
  return Cyc + Delta;
}

/// Helper class to simplify emission of cycle counts into optimization remarks.
struct Cycles {
  const char *Key;
  unsigned Value;
};
template <typename Remark> Remark &operator<<(Remark &R, Cycles C) {
  return R << ore::NV(C.Key, C.Value) << (C.Value == 1 ? " cycle" : " cycles");
}

struct SpeculateStrategy : SSAIfConv::PredicationStrategyBase {
  MachineLoopInfo *Loops = nullptr;
  const MCSchedModel &SchedModel;
  MachineTraceMetrics *Traces = nullptr;

  SpeculateStrategy(MachineLoopInfo *Loops, const MCSchedModel &SchedModel,
                    MachineTraceMetrics *Traces = nullptr)
      : Loops(Loops), SchedModel(SchedModel), Traces(Traces) {}

  bool canConvertIf(MachineBasicBlock *Head, MachineBasicBlock *TBB,
                    MachineBasicBlock *FBB, MachineBasicBlock *Tail,
                    ArrayRef<MachineOperand> Cond) override {
    // This is a triangle or a diamond.
    // Skip if we cannot predicate and there are no phis skip as there must
    // be side effects that can only be handled with predication.
    if (Tail->empty() || !Tail->front().isPHI()) {
      LLVM_DEBUG(dbgs() << "No phis in tail.\n");
      return false;
    }
    return true;
  }

  bool canPredicateInstr(const MachineInstr &I) override {
    // Don't speculate loads. Note that it may be possible and desirable to
    // speculate GOT or constant pool loads that are guaranteed not to trap,
    // but we don't support that for now.
    if (I.mayLoad()) {
      LLVM_DEBUG(dbgs() << "Won't speculate load: " << I);
      return false;
    }

    // We never speculate stores, so an AA pointer isn't necessary.
    bool DontMoveAcrossStore = true;
    if (!I.isSafeToMove(DontMoveAcrossStore)) {
      LLVM_DEBUG(dbgs() << "Can't speculate: " << I);
      return false;
    }
    return true;
  }

  bool shouldConvertIf(SSAIfConv &IfConv) override;

  void predicateBlock(MachineBasicBlock *MBB, ArrayRef<MachineOperand> Cond,
                      bool Reverse)
      override { /* do nothing, everything is speculatable and it's valid to
                    move the instructions into the head */
  }

  ~SpeculateStrategy() override = default;
};

bool SpeculateStrategy::shouldConvertIf(SSAIfConv &IfConv) {
  // Do not try to if-convert if the condition has a high chance of being
  // predictable.
  MachineLoop *CurrentLoop = Loops->getLoopFor(IfConv.Head);
  // If the condition is in a loop, consider it predictable if the condition
  // itself or all its operands are loop-invariant. E.g. this considers a load
  // from a loop-invariant address predictable; we were unable to prove that it
  // doesn't alias any of the memory-writes in the loop, but it is likely to
  // read to same value multiple times.
  if (CurrentLoop && any_of(IfConv.Cond, [&](MachineOperand &MO) {
        if (!MO.isReg() || !MO.isUse())
          return false;
        Register Reg = MO.getReg();
        if (Register::isPhysicalRegister(Reg))
          return false;

        MachineRegisterInfo *MRI = &IfConv.Head->getParent()->getRegInfo();
        MachineInstr *Def = MRI->getVRegDef(Reg);
        return CurrentLoop->isLoopInvariant(*Def) ||
               all_of(Def->operands(), [&](MachineOperand &Op) {
                 if (Op.isImm())
                   return true;
                 if (!MO.isReg() || !MO.isUse())
                   return false;
                 Register Reg = MO.getReg();
                 if (Register::isPhysicalRegister(Reg))
                   return false;

                 MachineInstr *Def = MRI->getVRegDef(Reg);
                 return CurrentLoop->isLoopInvariant(*Def);
               });
      }))
    return false;

  auto *MinInstr = IfConv.getEnsemble(MachineTraceStrategy::TS_MinInstrCount);
  assert(MinInstr);

  MachineTraceMetrics::Trace TBBTrace = MinInstr->getTrace(IfConv.getTPred());
  MachineTraceMetrics::Trace FBBTrace = MinInstr->getTrace(IfConv.getFPred());
  LLVM_DEBUG(dbgs() << "TBB: " << TBBTrace << "FBB: " << FBBTrace);
  unsigned MinCrit = std::min(TBBTrace.getCriticalPath(),
                              FBBTrace.getCriticalPath());

  // Set a somewhat arbitrary limit on the critical path extension we accept.
  unsigned CritLimit = SchedModel.MispredictPenalty/2;

  MachineBasicBlock &MBB = *IfConv.Head;
  MachineOptimizationRemarkEmitter MORE(*MBB.getParent(), nullptr);

  // If-conversion only makes sense when there is unexploited ILP. Compute the
  // maximum-ILP resource length of the trace after if-conversion. Compare it
  // to the shortest critical path.
  SmallVector<const MachineBasicBlock*, 1> ExtraBlocks;
  if (IfConv.TBB != IfConv.Tail)
    ExtraBlocks.push_back(IfConv.TBB);
  unsigned ResLength = FBBTrace.getResourceLength(ExtraBlocks);
  LLVM_DEBUG(dbgs() << "Resource length " << ResLength
                    << ", minimal critical path " << MinCrit << '\n');
  if (ResLength > MinCrit + CritLimit) {
    LLVM_DEBUG(dbgs() << "Not enough available ILP.\n");
    MORE.emit([&]() {
      MachineOptimizationRemarkMissed R(DEBUG_TYPE, "IfConversion",
                                        MBB.findDebugLoc(MBB.back()), &MBB);
      R << "did not if-convert branch: the resulting critical path ("
        << Cycles{"ResLength", ResLength}
        << ") would extend the shorter leg's critical path ("
        << Cycles{"MinCrit", MinCrit} << ") by more than the threshold of "
        << Cycles{"CritLimit", CritLimit}
        << ", which cannot be hidden by available ILP.";
      return R;
    });
    return false;
  }

  // Assume that the depth of the first head terminator will also be the depth
  // of the select instruction inserted, as determined by the flag dependency.
  // TBB / FBB data dependencies may delay the select even more.
  MachineTraceMetrics::Trace HeadTrace = MinInstr->getTrace(IfConv.Head);
  unsigned BranchDepth =
      HeadTrace.getInstrCycles(*IfConv.Head->getFirstTerminator()).Depth;
  LLVM_DEBUG(dbgs() << "Branch depth: " << BranchDepth << '\n');

  // Look at all the tail phis, and compute the critical path extension caused
  // by inserting select instructions.
  MachineTraceMetrics::Trace TailTrace = MinInstr->getTrace(IfConv.Tail);
  struct CriticalPathInfo {
    unsigned Extra; // Count of extra cycles that the component adds.
    unsigned Depth; // Absolute depth of the component in cycles.
  };
  CriticalPathInfo Cond{};
  CriticalPathInfo TBlock{};
  CriticalPathInfo FBlock{};
  bool ShouldConvert = true;
  for (SSAIfConv::PHIInfo &PI : IfConv.PHIs) {
    unsigned Slack = TailTrace.getInstrSlack(*PI.PHI);
    unsigned MaxDepth = Slack + TailTrace.getInstrCycles(*PI.PHI).Depth;
    LLVM_DEBUG(dbgs() << "Slack " << Slack << ":\t" << *PI.PHI);

    // The condition is pulled into the critical path.
    unsigned CondDepth = adjCycles(BranchDepth, PI.CondCycles);
    if (CondDepth > MaxDepth) {
      unsigned Extra = CondDepth - MaxDepth;
      LLVM_DEBUG(dbgs() << "Condition adds " << Extra << " cycles.\n");
      if (Extra > Cond.Extra)
        Cond = {Extra, CondDepth};
      if (Extra > CritLimit) {
        LLVM_DEBUG(dbgs() << "Exceeds limit of " << CritLimit << '\n');
        ShouldConvert = false;
      }
    }

    // The TBB value is pulled into the critical path.
    unsigned TDepth = adjCycles(TBBTrace.getPHIDepth(*PI.PHI), PI.TCycles);
    if (TDepth > MaxDepth) {
      unsigned Extra = TDepth - MaxDepth;
      LLVM_DEBUG(dbgs() << "TBB data adds " << Extra << " cycles.\n");
      if (Extra > TBlock.Extra)
        TBlock = {Extra, TDepth};
      if (Extra > CritLimit) {
        LLVM_DEBUG(dbgs() << "Exceeds limit of " << CritLimit << '\n');
        ShouldConvert = false;
      }
    }

    // The FBB value is pulled into the critical path.
    unsigned FDepth = adjCycles(FBBTrace.getPHIDepth(*PI.PHI), PI.FCycles);
    if (FDepth > MaxDepth) {
      unsigned Extra = FDepth - MaxDepth;
      LLVM_DEBUG(dbgs() << "FBB data adds " << Extra << " cycles.\n");
      if (Extra > FBlock.Extra)
        FBlock = {Extra, FDepth};
      if (Extra > CritLimit) {
        LLVM_DEBUG(dbgs() << "Exceeds limit of " << CritLimit << '\n');
        ShouldConvert = false;
      }
    }
  }

  // Organize by "short" and "long" legs, since the diagnostics get confusing
  // when referring to the "true" and "false" sides of the branch, given that
  // those don't always correlate with what the user wrote in source-terms.
  const CriticalPathInfo Short = TBlock.Extra > FBlock.Extra ? FBlock : TBlock;
  const CriticalPathInfo Long = TBlock.Extra > FBlock.Extra ? TBlock : FBlock;

  if (ShouldConvert) {
    MORE.emit([&]() {
      MachineOptimizationRemark R(DEBUG_TYPE, "IfConversion",
                                  MBB.back().getDebugLoc(), &MBB);
      R << "performing if-conversion on branch: the condition adds "
        << Cycles{"CondCycles", Cond.Extra} << " to the critical path";
      if (Short.Extra > 0)
        R << ", and the short leg adds another "
          << Cycles{"ShortCycles", Short.Extra};
      if (Long.Extra > 0)
        R << ", and the long leg adds another "
          << Cycles{"LongCycles", Long.Extra};
      R << ", each staying under the threshold of "
        << Cycles{"CritLimit", CritLimit} << ".";
      return R;
    });
  } else {
    MORE.emit([&]() {
      MachineOptimizationRemarkMissed R(DEBUG_TYPE, "IfConversion",
                                        MBB.back().getDebugLoc(), &MBB);
      R << "did not if-convert branch: the condition would add "
        << Cycles{"CondCycles", Cond.Extra} << " to the critical path";
      if (Cond.Extra > CritLimit)
        R << " exceeding the limit of " << Cycles{"CritLimit", CritLimit};
      if (Short.Extra > 0) {
        R << ", and the short leg would add another "
          << Cycles{"ShortCycles", Short.Extra};
        if (Short.Extra > CritLimit)
          R << " exceeding the limit of " << Cycles{"CritLimit", CritLimit};
      }
      if (Long.Extra > 0) {
        R << ", and the long leg would add another "
          << Cycles{"LongCycles", Long.Extra};
        if (Long.Extra > CritLimit)
          R << " exceeding the limit of " << Cycles{"CritLimit", CritLimit};
      }
      R << ".";
      return R;
    });
  }

  return ShouldConvert;
}

bool EarlyIfConverter::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** EARLY IF-CONVERSION **********\n"
                    << "********** Function: " << MF.getName() << '\n');
  if (skipFunction(MF.getFunction()))
    return false;

  // Only run if conversion if the target wants it.
  const TargetSubtargetInfo &STI = MF.getSubtarget();
  if (!STI.enableEarlyIfConversion())
    return false;

  const MCSchedModel &SchedModel = STI.getSchedModel();
  auto *DomTree = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto *Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineTraceMetrics *Traces = &getAnalysis<MachineTraceMetrics>();

  SpeculateStrategy Speculate(Loops, SchedModel);
  SSAIfConv IfConv(Speculate, MF, DomTree, Loops, Traces);
  return IfConv.run();
}
} // end anonymous namespace

char &llvm::EarlyIfConverterID = EarlyIfConverter::ID;

INITIALIZE_PASS_BEGIN(EarlyIfConverter, DEBUG_TYPE, "Early If Converter", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineTraceMetrics)
INITIALIZE_PASS_END(EarlyIfConverter, DEBUG_TYPE, "Early If Converter", false,
                    false)
#undef DEBUG_TYPE

#define DEBUG_TYPE "early-if-predicator"
namespace {
struct EarlyIfPredicator : MachineFunctionPass {
  static char ID;
  EarlyIfPredicator() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return "Early If-predicator"; }
};

char EarlyIfPredicator::ID = 0;

void EarlyIfPredicator::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

struct PredicatorStrategy : SSAIfConv::PredicationStrategyBase {
  const TargetInstrInfo *TII = nullptr;
  TargetSchedModel &SchedModel;
  MachineBranchProbabilityInfo *MBPI = nullptr;
  PredicatorStrategy(const TargetInstrInfo *TII, TargetSchedModel &SchedModel,
                     MachineBranchProbabilityInfo *MBPI)
      : TII(TII), SchedModel(SchedModel), MBPI(MBPI) {}

  bool canPredicateInstr(const MachineInstr &I) override {
    // Check that instruction is predicable
    if (!TII->isPredicable(I)) {
      LLVM_DEBUG(dbgs() << "Isn't predicable: " << I);
      return false;
    }

    // Check that instruction is not already predicated.
    if (TII->isPredicated(I) && !TII->canPredicatePredicatedInstr(I)) {
      LLVM_DEBUG(dbgs() << "Is already predicated: " << I);
      return false;
    }
    return true;
  }

  bool shouldConvertIf(SSAIfConv &IfConv) override;

  void predicateBlock(MachineBasicBlock *MBB, ArrayRef<MachineOperand> Cond,
                      bool Reverse) override {
    SmallVector<MachineOperand> Condition(Cond.begin(), Cond.end());
    if (Reverse) {
      bool CanRevCond = !TII->reverseBranchCondition(Condition);
      assert(CanRevCond && "Reversed predicate is not supported");
      (void)CanRevCond;
    }
    // Terminators don't need to be predicated as they will be removed.
    for (MachineBasicBlock::iterator I = MBB->begin(),
                                     E = MBB->getFirstTerminator();
         I != E; ++I) {
      if (I->isDebugInstr())
        continue;
      TII->PredicateInstruction(*I, Condition);
    }
  }

  ~PredicatorStrategy() override = default;
};

bool PredicatorStrategy::shouldConvertIf(SSAIfConv &IfConv) {
  auto TrueProbability = MBPI->getEdgeProbability(IfConv.Head, IfConv.TBB);
  if (IfConv.isTriangle()) {
    MachineBasicBlock &IfBlock =
        (IfConv.TBB == IfConv.Tail) ? *IfConv.FBB : *IfConv.TBB;

    unsigned ExtraPredCost = 0;
    unsigned Cycles = 0;
    for (MachineInstr &I : IfBlock) {
      unsigned NumCycles = SchedModel.computeInstrLatency(&I, false);
      if (NumCycles > 1)
        Cycles += NumCycles - 1;
      ExtraPredCost += TII->getPredicationCost(I);
    }

    return TII->isProfitableToIfCvt(IfBlock, Cycles, ExtraPredCost,
                                    TrueProbability);
  }
  unsigned TExtra = 0;
  unsigned FExtra = 0;
  unsigned TCycle = 0;
  unsigned FCycle = 0;
  for (MachineInstr &I : *IfConv.TBB) {
    unsigned NumCycles = SchedModel.computeInstrLatency(&I, false);
    if (NumCycles > 1)
      TCycle += NumCycles - 1;
    TExtra += TII->getPredicationCost(I);
  }
  for (MachineInstr &I : *IfConv.FBB) {
    unsigned NumCycles = SchedModel.computeInstrLatency(&I, false);
    if (NumCycles > 1)
      FCycle += NumCycles - 1;
    FExtra += TII->getPredicationCost(I);
  }
  return TII->isProfitableToIfCvt(*IfConv.TBB, TCycle, TExtra, *IfConv.FBB,
                                  FCycle, FExtra, TrueProbability);
}

bool EarlyIfPredicator::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** EARLY IF-PREDICATOR **********\n"
                    << "********** Function: " << MF.getName() << '\n');
  if (skipFunction(MF.getFunction()))
    return false;

  const TargetSubtargetInfo &STI = MF.getSubtarget();
  auto *TII = STI.getInstrInfo();
  TargetSchedModel SchedModel;
  SchedModel.init(&STI);
  auto *DomTree = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto *Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  auto *MBPI =
      &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();

  PredicatorStrategy Predicate(TII, SchedModel, MBPI);
  SSAIfConv IfConv(Predicate, MF, DomTree, Loops);
  return IfConv.run();
}

} // end anonymous namespace
char &llvm::EarlyIfPredicatorID = EarlyIfPredicator::ID;

INITIALIZE_PASS_BEGIN(EarlyIfPredicator, DEBUG_TYPE, "Early If Predicator",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(EarlyIfPredicator, DEBUG_TYPE, "Early If Predicator", false,
                    false)