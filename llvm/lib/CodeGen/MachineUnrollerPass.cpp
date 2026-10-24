//===- MachineUnrollerPass.cpp - Machine loop unroller pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements loop unrolling functionality at the machine instruction
// (MI) level.
// This pass complements IR-level loop unrolling rather than replacing it.
// The machine unroller runs on MachineInstrs, where it has access to target
// specific information such as instruction latencies, resource usage, etc.
// Using this information, it can identify loops where unrolling will actually
// increase resource utilization and skip those where it would not. Combined
// with the software pipeliner, this can significantly improve the performance
// of certain loops on VLIW-type architectures.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineUnroller.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "machine-unroller"

using NV = DiagnosticInfoOptimizationBase::Argument;

/// A command line option to turn MI Loop Unrolling on or off.
static cl::opt<bool> EnableMIUnroller("enable-machine-unroller", cl::Hidden,
                                      cl::init(true), cl::ZeroOrMore,
                                      cl::desc("Enable MI Loop Unrolling"));

/// A command line argument to limit size of the unrolled loop.
static cl::opt<unsigned>
    MachineUnrollerThres("machine-unroller-threshold",
                         cl::desc("Size limit for the unrolled loop."),
                         cl::Hidden, cl::init(30));

/// A command line option to enable MI Loop Unrolling at -Os.
static cl::opt<bool>
    EnableMIUnrollerOptSize("enable-machine-unroller-opt-size",
                            cl::desc("Enable MI Loop Unrolling at Os."),
                            cl::Hidden, cl::init(false));

#ifndef NDEBUG
static cl::opt<int> UnrollerLimit("machine-unroller-max", cl::Hidden,
                                  cl::init(-1));
#endif

typedef std::set<MachineInstr *> MISet;

namespace {
class MachineUnrollerPass : public MachineUnrollerContext,
                            public MachineFunctionPass {
  const TargetPassConfig *PassConfig = nullptr;
  MachineUnroller *Unroller = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineOptimizationRemarkEmitter *ORE = nullptr;
  bool tryToUnrollLoop(MachineLoop &L);
  bool unrollLoop(MachineLoop *L, unsigned UnrollFactor);
  bool canUnrollLoop(MachineLoop *L);

public:
  static char ID;
  MachineUnrollerPass() : MachineFunctionPass(ID) {
    initializeMachineUnrollerPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
#ifndef NDEBUG
  static int NumTries;
#endif
};

} // end anonymous namespace

char MachineUnrollerPass::ID = 0;
#ifndef NDEBUG
int MachineUnrollerPass::NumTries = 0;
#endif
char &llvm::MachineUnrollerPassID = MachineUnrollerPass::ID;
INITIALIZE_PASS_BEGIN(MachineUnrollerPass, DEBUG_TYPE, "Machine Unrolling",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(MachineUnrollerPass, DEBUG_TYPE, "Machine Unrolling", false,
                    false)

class MachineUnrollerSchedDAG : public ScheduleDAGInstrs {
  MachineUnrollerPass &Pass;
  MachineLoop &Loop;
  MachineOptimizationRemarkEmitter *ORE;

public:
  MachineUnrollerSchedDAG(MachineUnrollerPass &P, MachineLoop &L,
                          MachineOptimizationRemarkEmitter *ORE)
      : ScheduleDAGInstrs(*P.MF, P.MLI, false), Pass(P), Loop(L), ORE(ORE) {};
  unsigned getUnrollFactor();
  void schedule() override;

private:
  unsigned calculateResMII(unsigned UnrollFactor);
  unsigned adjustResMIIForExtraCopies(unsigned ResMII);
  bool shouldNotUnroll(MachineLoop &Loop, int MinResMII, MISet &selfDepInstr);
  SmallVector<DFAPacketizer *, 8> Resources;
};

// FuncUnitSorter - Comparison operator used to sort instructions by
// the number of functional unit choices.
struct FuncUnitSorter {
  const InstrItineraryData *InstrItins;
  DenseMap<unsigned, unsigned> Resources;

  FuncUnitSorter(const InstrItineraryData *IID) : InstrItins(IID) {}

  // Compute the number of functional unit alternatives needed
  // at each stage, and take the minimum value. We prioritize the
  // instructions by the least number of choices first.
  unsigned minFuncUnits(const MachineInstr *Inst, unsigned &F) const {
    unsigned schedClass = Inst->getDesc().getSchedClass();
    unsigned min = UINT_MAX;
    for (const InstrStage *IS = InstrItins->beginStage(schedClass),
                          *IE = InstrItins->endStage(schedClass);
         IS != IE; ++IS) {
      unsigned funcUnits = IS->getUnits();
      unsigned numAlternatives = llvm::popcount(funcUnits);
      if (numAlternatives < min) {
        min = numAlternatives;
        F = funcUnits;
      }
    }
    return min;
  }

  // Compute the critical resources needed by the instruction. This
  // function records the functional units needed by instructions that
  // must use only one functional unit. We use this as a tie breaker
  // for computing the resource MII. The instrutions that require
  // the same, highly used, functional unit have high priority.
  void calcCriticalResources(MachineInstr &MI) {
    unsigned SchedClass = MI.getDesc().getSchedClass();
    for (const InstrStage *IS = InstrItins->beginStage(SchedClass),
                          *IE = InstrItins->endStage(SchedClass);
         IS != IE; ++IS) {
      unsigned FuncUnits = IS->getUnits();
      if (llvm::popcount(FuncUnits) == 1)
        Resources[FuncUnits]++;
    }
  }

  /// Return true if IS1 has less priority than IS2.
  bool operator()(const MachineInstr *IS1, const MachineInstr *IS2) const {
    unsigned F1 = 0, F2 = 0;
    unsigned MFUs1 = minFuncUnits(IS1, F1);
    unsigned MFUs2 = minFuncUnits(IS2, F2);
    if (MFUs1 == 1 && MFUs2 == 1)
      return Resources.lookup(F1) < Resources.lookup(F2);
    return MFUs1 > MFUs2;
  }
};

bool MachineUnrollerPass::tryToUnrollLoop(MachineLoop &L) {
  bool Changed = false;
  for (auto &InnerLoop : L)
    Changed |= tryToUnrollLoop(*InnerLoop);

#ifndef NDEBUG
  // Stop trying after reaching the limit (if any).
  int Limit = UnrollerLimit;
  if (Limit >= 0) {
    if (NumTries >= UnrollerLimit)
      return Changed;
    NumTries++;
  }
#endif

  if (!canUnrollLoop(&L))
    return Changed;

  Changed = unrollLoop(&L, 1);
  return Changed;
}

bool MachineUnrollerPass::canUnrollLoop(MachineLoop *L) {
  // Only loops with a single basic block are handled. Also, the loop must
  // be analyzable using analyzeBranch. It's the responsibility of the caller of
  // this function to make sure that these requirement are met.
  if (L->getNumBlocks() > 1) {
    LLVM_DEBUG(
        dbgs() << "Only loops with single basic block can be unrolled!!");
    return false;
  }

  return true;
}

bool MachineUnrollerPass::unrollLoop(MachineLoop *L, unsigned UnrollFactor) {
  MachineUnrollerSchedDAG MSD(*this, *L, ORE);

  MachineBasicBlock *MBB = L->getHeader();
  // The kernel should not include any terminator instructions.  These
  // will be added back later.
  MSD.startBlock(MBB);
  unsigned size = MBB->size();
  for (MachineBasicBlock::iterator I = MBB->getFirstTerminator(),
                                   E = MBB->instr_end();
       I != E; ++I, --size)
    ;

  MSD.enterRegion(MBB, MBB->begin(), MBB->getFirstTerminator(), size);
  MSD.schedule();
  UnrollFactor = MSD.getUnrollFactor();
  bool Changed = false;
  if (UnrollFactor > 1)
    Changed = Unroller->unroll(L, UnrollFactor);
  MSD.exitRegion();
  return Changed;
}

void MachineUnrollerSchedDAG::schedule() {
  AliasAnalysis *AA = &Pass.getAnalysis<AAResultsWrapperPass>().getAAResults();
  buildSchedGraph(AA);
}

static unsigned getNonDebugMBBSize(MachineBasicBlock *MBB) {
  int size = 0;
  for (MachineBasicBlock::iterator I = MBB->getFirstNonPHI(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I) {
    if (!I->isDebugInstr())
      size++;
  }
  return size;
}

// Check if their is a self register dependence between same instruction across
// iterations.
void checkSelfDependence(MachineLoop &Loop, MISet &selfDepInstr) {
  MachineBasicBlock *MBB = Loop.getHeader();
  // Track Register Dependencies from PHI to Inst or from Inst to PHI
  std::map<std::pair<MachineInstr *, MachineInstr *>, bool> deps;
  // Registers defined by PHI Node
  std::map<Register, MachineInstr *> phiDefs;
  // Registers used by PHI Node
  std::map<Register, std::vector<MachineInstr *>> phiUses;
  // Populate phiDefs and phiUses
  for (MachineBasicBlock::iterator I = MBB->instr_begin(),
                                   E = MBB->getFirstNonPHI();
       I != E; ++I) {
    for (MachineOperand MO : I->operands()) {
      if (!MO.isReg())
        continue;
      if (MO.isDef())
        phiDefs[MO.getReg()] = &*I;
      else
        phiUses[MO.getReg()].push_back(&*I);
    }
  }
  // Self Dependency: Check for Instructions which define an operand used by
  // PHI node and use an operand defined by PHI Node
  for (MachineBasicBlock::iterator I = MBB->getFirstNonPHI(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I) {
    for (MachineOperand MO : I->operands()) {
      if (!MO.isReg())
        continue;
      Register r = MO.getReg();
      if (MO.isUse() && phiDefs.find(r) != phiDefs.end()) {
        // Edge from PHI to Instruction
        deps[{phiDefs[r], &*I}] = true;
        if (deps.find({&*I, phiDefs[r]}) != deps.end())
          selfDepInstr.insert(&*I);
      } else if (MO.isDef() && phiUses.find(r) != phiUses.end()) {
        // Edge from Instruction to PHI
        for (MachineInstr *phi : phiUses[r]) {
          deps[{&*I, phi}] = true;
          if (deps.find({phi, &*I}) != deps.end())
            selfDepInstr.insert(&*I);
        }
      }
    }
  }
}

// Do not unroll if the following conditions are true:
// 1. There exists a self dependent instruction with latency >= MinResMII.
// 2. No non-self dependent instruction has latency > MinResMII.
// 3. Atleast half of the instructions in the loop are independent.
bool MachineUnrollerSchedDAG::shouldNotUnroll(MachineLoop &Loop, int MinResMII,
                                              MISet &selfDepInstr) {
  MachineBasicBlock *MBB = Loop.getHeader();
  MachineFunction *MF = MBB->getParent();
  const InstrItineraryData *InstrItins =
      MF->getSubtarget().getInstrItineraryData();
  std::map<MachineInstr *, int> Latencies;
  bool ShouldNotUnroll = false;
  unsigned NonPhiInst = 0;
  // Calculate latency for each instruction in the loop.
  for (MachineBasicBlock::iterator I = MBB->getFirstNonPHI(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I) {
    LLVM_DEBUG({
      dbgs() << "Instr = ";
      I->dump();
    });

    if (I->isDebugInstr())
      continue;
    // Find the latency of each use-operand in the instruction.
    NonPhiInst++;
    for (MachineOperand MO : I->uses()) {
      if (!MO.isReg() || MO.isImplicit() || MO.getReg().isPhysical())
        continue;
      MachineInstr *MIDef = Pass.MF->getRegInfo().getVRegDef(MO.getReg());
      if (!MIDef)
        continue;
      int RegDefIdx =
          MIDef->findRegisterDefOperandIdx(MO.getReg(), /*TRI=*/nullptr);
      int RegUseIdx =
          I->findRegisterUseOperandIdx(MO.getReg(), /*TRI=*/nullptr);
      std::optional<unsigned> Curr =
          TII->getOperandLatency(InstrItins, *MIDef, RegDefIdx, *I, RegUseIdx);
      // If the latency calculated is null then set the operator latency to 1 or
      // retain the value if it already exists.
      if (Latencies.find(&*I) == Latencies.end()) {
        if (Curr.has_value()) {
          Latencies[&*I] = std::max((int)(*Curr), 1);
        } else {
          Latencies[&*I] = 1;
        }
      } else if (Curr.has_value()) {
        Latencies[&*I] = std::max(Latencies[&*I], (int)(*Curr));
      }

      LLVM_DEBUG({
        dbgs() << "\tOperand = ";
        MO.print(dbgs());
        dbgs() << ",\tLatency = " << Latencies[&*I] << "\n";
      });

      if (Latencies[&*I] >= MinResMII) {
        if (selfDepInstr.find(&*I) != selfDepInstr.end())
          // Self dependent instruction with latency >= MinResMII and
          // atleast half of the instructions in the loop are independent.
          ShouldNotUnroll |= ((Latencies[&*I] >= MinResMII) &&
                              (selfDepInstr.size() * 2 > NonPhiInst));
        else {
          // No non-self dependent instruction should have latency > MinResMII.
          return false;
        }
      }
    }
  }
  return ShouldNotUnroll;
}

unsigned MachineUnrollerSchedDAG::getUnrollFactor() {
  unsigned InitialResMII = calculateResMII(1);
  InitialResMII = adjustResMIIForExtraCopies(InitialResMII);
  unsigned MinResMII = InitialResMII;
  unsigned MinUnrollFactor = 1;
  unsigned UnrollThres = 4;
  unsigned LoopHeaderSize = getNonDebugMBBSize(Loop.getHeader());

  // Check for instruction self dependencies
  MISet selfDepInstr;
  checkSelfDependence(Loop, selfDepInstr);
  LLVM_DEBUG(dbgs() << "Self Dependent Inst count = " << selfDepInstr.size()
                    << "\n");
  if (shouldNotUnroll(Loop, MinResMII, selfDepInstr)) {
    LLVM_DEBUG(dbgs() << "Self Dependencies Found. Using unroll factor = 1\n");
    ORE->emit([&]() {
      return MachineOptimizationRemarkMissed(
                 DEBUG_TYPE, "SelfDependency",
                 Loop.getHeader()->front().getDebugLoc(), Loop.getHeader())
             << "Unable to unroll loop: self dependencies found";
    });
    return 1;
  }

  bool AnyBenefit = false;
  bool AllBeneficialExceededThreshold = true;
  for (unsigned i = 2; i <= UnrollThres; i += 2) {
    unsigned UnrollResMII = calculateResMII(i);
    LLVM_DEBUG(dbgs() << "Unroll Factor = " << i << "(res=" << UnrollResMII
                      << ")\n");
    float UnrollResMIIRatio = (float)UnrollResMII / i;
    float MinResMIIRatio = (float)MinResMII / MinUnrollFactor;

    if (UnrollResMIIRatio < MinResMIIRatio) {
      AnyBenefit = true;
      if ((LoopHeaderSize * i) <= MachineUnrollerThres) {
        AllBeneficialExceededThreshold = false;
        MinResMII = UnrollResMII;
        MinUnrollFactor = i;
      } else {
        LLVM_DEBUG(dbgs() << "Loop size " << (LoopHeaderSize * i)
                          << " exceeds threshold " << MachineUnrollerThres
                          << " for factor " << i << "\n");
        ORE->emit([&]() {
          return MachineOptimizationRemarkMissed(
                     DEBUG_TYPE, "SizeLimit",
                     Loop.getHeader()->front().getDebugLoc(), Loop.getHeader())
                 << "Unable to unroll loop by factor " << NV("Factor", i)
                 << ": unrolled size " << NV("Size", LoopHeaderSize * i)
                 << " exceeds threshold "
                 << NV("Threshold", (unsigned)MachineUnrollerThres);
        });
      }
    } else {
      LLVM_DEBUG(dbgs() << "Unroll factor " << i
                        << " did not improve ResMII\n");
    }
  }

  if (MinUnrollFactor > 1) {
    ORE->emit([&]() {
      return MachineOptimizationRemark(DEBUG_TYPE, "Unrolled",
                                       Loop.getHeader()->front().getDebugLoc(),
                                       Loop.getHeader())
             << "Unrolled loop by factor "
             << NV("UnrollFactor", MinUnrollFactor) << " (ResMII improved from "
             << NV("InitialResMII", InitialResMII) << " to "
             << NV("FinalResMII", MinResMII) << ")";
    });
  } else if (AnyBenefit && AllBeneficialExceededThreshold) {
    ORE->emit([&]() {
      return MachineOptimizationRemarkMissed(
                 DEBUG_TYPE, "AllExceededThreshold",
                 Loop.getHeader()->front().getDebugLoc(), Loop.getHeader())
             << "Unable to unroll loop: all beneficial factors exceeded size "
                "threshold";
    });
  } else if (!AnyBenefit) {
    ORE->emit([&]() {
      return MachineOptimizationRemarkMissed(
                 DEBUG_TYPE, "NoBenefit",
                 Loop.getHeader()->front().getDebugLoc(), Loop.getHeader())
             << "Unable to unroll loop: unrolling does not improve ResMII";
    });
  }

  LLVM_DEBUG(dbgs() << "Using unroll factor of " << MinUnrollFactor << "\n");
  return MinUnrollFactor;
}

unsigned MachineUnrollerSchedDAG::calculateResMII(unsigned UnrollFactor) {
  SmallVector<DFAPacketizer *, 8> Resources;
  MachineBasicBlock *MBB = Loop.getHeader();
  Resources.push_back(TII->CreateTargetScheduleState(MF.getSubtarget()));

  // Sort the instructions by the number of available choices for scheduling,
  // least to most. Use the number of critical resources as the tie breaker.
  FuncUnitSorter FUS =
      FuncUnitSorter(MF.getSubtarget().getInstrItineraryData());
  for (MachineBasicBlock::iterator I = MBB->getFirstNonPHI(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I)
    FUS.calcCriticalResources(*I);
  PriorityQueue<MachineInstr *, std::vector<MachineInstr *>, FuncUnitSorter>
      FuncUnitOrder(FUS);

  // To compute ResMII for the unrolled loop, simply replicate instructions as
  // many times as the unroll factor.
  for (unsigned i = 0; i < UnrollFactor; i++) {
    for (MachineBasicBlock::iterator I = MBB->getFirstNonPHI(),
                                     E = MBB->getFirstTerminator();
         I != E; ++I)
      FuncUnitOrder.push(&*I);
  }
  while (!FuncUnitOrder.empty()) {
    MachineInstr *MI = FuncUnitOrder.top();
    FuncUnitOrder.pop();
    if (TII->isZeroCost(MI->getOpcode()))
      continue;
    // Attempt to reserve the instruction in an existing DFA. At least one
    // DFA is needed for each cycle.
    unsigned NumCycles = 1;
    unsigned ReservedCycles = 0;
    SmallVectorImpl<DFAPacketizer *>::iterator RI = Resources.begin();
    SmallVectorImpl<DFAPacketizer *>::iterator RE = Resources.end();
    for (unsigned C = 0; C < NumCycles; ++C)
      while (RI != RE) {
        if ((*RI++)->canReserveResources(*MI)) {
          ++ReservedCycles;
          break;
        }
      }
    // Start reserving resources using existing DFAs.
    for (unsigned C = 0; C < ReservedCycles; ++C) {
      --RI;
      (*RI)->reserveResources(*MI);
    }
    // Add new DFAs, if needed, to reserve resources.
    for (unsigned C = ReservedCycles; C < NumCycles; ++C) {
      DFAPacketizer *NewResource =
          TII->CreateTargetScheduleState(MF.getSubtarget());
      assert(NewResource->canReserveResources(*MI) && "Reserve error.");
      NewResource->reserveResources(*MI);
      Resources.push_back(NewResource);
    }
  }
  int Resmii = Resources.size();
  // Delete the memory for each of the DFAs that were created earlier.
  for (DFAPacketizer *RI : Resources) {
    DFAPacketizer *D = RI;
    delete D;
  }
  Resources.clear();
  return Resmii;
}

/// Adjust starting ResMII if latency between any of the instructions in
/// the loop header happens to be higher than the previously computed value
/// which is passed as the input parameter. This is done to account for
/// the extra copies and therefore resources that are needed when the loop
/// is software pipelined later on. One thing to note here is that even if
/// the pipeliner is able to find a schedule with the original ResMII,
/// the high latencies between the instructions will always cause stalls.
/// Identifying such loops here and unrolling them can help the pipeliner
/// generate better schedule with fewer stalls.
unsigned MachineUnrollerSchedDAG::adjustResMIIForExtraCopies(unsigned ResMII) {
  unsigned MinResMII = ResMII;
  MachineBasicBlock *MBB = Loop.getHeader();
  for (auto &MI :
       make_range(MBB->getFirstNonPHI(), MBB->getFirstTerminator())) {
    if (MI.isDebugInstr())
      continue;
    SUnit *SU = getSUnit(&MI);
    for (auto &Dep : SU->Succs) {
      if (Dep.getSUnit() == SU)
        continue;
      if (Dep.getKind() != SDep::Data)
        continue;
      unsigned Latency = Dep.getLatency();
      if (Latency > MinResMII)
        MinResMII = Latency;
    }
  }
  return MinResMII;
}

bool MachineUnrollerPass::runOnMachineFunction(MachineFunction &mf) {
  if (skipFunction(mf.getFunction()))
    return false;

  if (!EnableMIUnroller)
    return false;

  if (mf.getFunction().getAttributes().hasAttributeAtIndex(
          AttributeList::FunctionIndex, Attribute::OptimizeForSize) &&
      !EnableMIUnrollerOptSize.getPosition())
    return false;

  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  TII = MF->getSubtarget().getInstrInfo();
  MRI = &MF->getRegInfo();
  PassConfig = &getAnalysis<TargetPassConfig>();
  ORE = &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
  Unroller = PassConfig->createMachineUnroller(this);
  if (!Unroller)
    return false;

  bool Changed = false;
  for (auto &L : *MLI)
    Changed |= tryToUnrollLoop(*L);

  delete Unroller;
  return Changed;
}
