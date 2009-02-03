//===----- SchedulePostRAList.cpp - list scheduler ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a top-down list scheduler, using standard algorithms.
// The basic approach uses a priority queue of available nodes to schedule.
// One at a time, nodes are taken from the priority queue (thus in priority
// order), checked for legality to schedule, and emitted if legal.
//
// Nodes may not be legal to schedule either due to structural hazards (e.g.
// pipeline or resource constraints) or because an input to the instruction has
// not completed execution.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "post-RA-sched"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <map>
using namespace llvm;

STATISTIC(NumNoops, "Number of noops inserted");
STATISTIC(NumStalls, "Number of pipeline stalls");

static cl::opt<bool>
EnableAntiDepBreaking("break-anti-dependencies",
                      cl::desc("Break post-RA scheduling anti-dependencies"),
                      cl::init(true), cl::Hidden);

static cl::opt<bool>
EnablePostRAHazardAvoidance("avoid-hazards",
                      cl::desc("Enable simple hazard-avoidance"),
                      cl::init(true), cl::Hidden);

namespace {
  class VISIBILITY_HIDDEN PostRAScheduler : public MachineFunctionPass {
  public:
    static char ID;
    PostRAScheduler() : MachineFunctionPass(&ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    const char *getPassName() const {
      return "Post RA top-down list latency scheduler";
    }

    bool runOnMachineFunction(MachineFunction &Fn);
  };
  char PostRAScheduler::ID = 0;

  class VISIBILITY_HIDDEN SchedulePostRATDList : public ScheduleDAGInstrs {
    /// AvailableQueue - The priority queue to use for the available SUnits.
    ///
    LatencyPriorityQueue AvailableQueue;
  
    /// PendingQueue - This contains all of the instructions whose operands have
    /// been issued, but their results are not ready yet (due to the latency of
    /// the operation).  Once the operands becomes available, the instruction is
    /// added to the AvailableQueue.
    std::vector<SUnit*> PendingQueue;

    /// Topo - A topological ordering for SUnits.
    ScheduleDAGTopologicalSort Topo;

    /// AllocatableSet - The set of allocatable registers.
    /// We'll be ignoring anti-dependencies on non-allocatable registers,
    /// because they may not be safe to break.
    const BitVector AllocatableSet;

    /// HazardRec - The hazard recognizer to use.
    ScheduleHazardRecognizer *HazardRec;

  public:
    SchedulePostRATDList(MachineFunction &MF,
                         const MachineLoopInfo &MLI,
                         const MachineDominatorTree &MDT,
                         ScheduleHazardRecognizer *HR)
      : ScheduleDAGInstrs(MF, MLI, MDT), Topo(SUnits),
        AllocatableSet(TRI->getAllocatableSet(MF)),
        HazardRec(HR) {}

    ~SchedulePostRATDList() {
      delete HazardRec;
    }

    void Schedule();

  private:
    void ReleaseSucc(SUnit *SU, SDep *SuccEdge);
    void ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle);
    void ListScheduleTopDown();
    bool BreakAntiDependencies();
  };

  /// SimpleHazardRecognizer - A *very* simple hazard recognizer. It uses
  /// a coarse classification and attempts to avoid that instructions of
  /// a given class aren't grouped too densely together.
  class SimpleHazardRecognizer : public ScheduleHazardRecognizer {
    /// Class - A simple classification for SUnits.
    enum Class {
      Other, Load, Store
    };

    /// Window - The Class values of the most recently issued
    /// instructions.
    Class Window[8];

    /// getClass - Classify the given SUnit.
    Class getClass(const SUnit *SU) {
      const MachineInstr *MI = SU->getInstr();
      const TargetInstrDesc &TID = MI->getDesc();
      if (TID.mayLoad())
        return Load;
      if (TID.mayStore())
        return Store;
      return Other;
    }

    /// Step - Rotate the existing entries in Window and insert the
    /// given class value in position as the most recent.
    void Step(Class C) {
      std::copy(Window+1, array_endof(Window), Window);
      Window[array_lengthof(Window)-1] = C;
    }

  public:
    SimpleHazardRecognizer() : Window() {}

    virtual HazardType getHazardType(SUnit *SU) {
      Class C = getClass(SU);
      if (C == Other)
        return NoHazard;
      unsigned Score = 0;
      for (unsigned i = 0; i != array_lengthof(Window); ++i)
        if (Window[i] == C)
          Score += i + 1;
      if (Score > array_lengthof(Window) * 2)
        return Hazard;
      return NoHazard;
    }

    virtual void EmitInstruction(SUnit *SU) {
      Step(getClass(SU));
    }

    virtual void AdvanceCycle() {
      Step(Other);
    }
  };
}

bool PostRAScheduler::runOnMachineFunction(MachineFunction &Fn) {
  DOUT << "PostRAScheduler\n";

  const MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  const MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();
  ScheduleHazardRecognizer *HR = EnablePostRAHazardAvoidance ?
                                 new SimpleHazardRecognizer :
                                 new ScheduleHazardRecognizer();

  SchedulePostRATDList Scheduler(Fn, MLI, MDT, HR);

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    // Schedule each sequence of instructions not interrupted by a label
    // or anything else that effectively needs to shut down scheduling.
    MachineBasicBlock::iterator Current = MBB->begin(), End = MBB->end();
    for (MachineBasicBlock::iterator MI = Current; MI != End; ++MI)
      if (MI->getDesc().isTerminator() || MI->isLabel()) {
        Scheduler.Run(0, MBB, Current, MI);
        Scheduler.EmitSchedule();
        Current = next(MI);
      }

    Scheduler.Run(0, MBB, Current, End);
    Scheduler.EmitSchedule();
  }

  return true;
}
  
/// Schedule - Schedule the DAG using list scheduling.
void SchedulePostRATDList::Schedule() {
  DOUT << "********** List Scheduling **********\n";
  
  // Build the scheduling graph.
  BuildSchedGraph();

  if (EnableAntiDepBreaking) {
    if (BreakAntiDependencies()) {
      // We made changes. Update the dependency graph.
      // Theoretically we could update the graph in place:
      // When a live range is changed to use a different register, remove
      // the def's anti-dependence *and* output-dependence edges due to
      // that register, and add new anti-dependence and output-dependence
      // edges based on the next live range of the register.
      SUnits.clear();
      BuildSchedGraph();
    }
  }

  AvailableQueue.initNodes(SUnits);

  ListScheduleTopDown();
  
  AvailableQueue.releaseState();
}

/// getInstrOperandRegClass - Return register class of the operand of an
/// instruction of the specified TargetInstrDesc.
static const TargetRegisterClass*
getInstrOperandRegClass(const TargetRegisterInfo *TRI,
                        const TargetInstrInfo *TII, const TargetInstrDesc &II,
                        unsigned Op) {
  if (Op >= II.getNumOperands())
    return NULL;
  if (II.OpInfo[Op].isLookupPtrRegClass())
    return TII->getPointerRegClass();
  return TRI->getRegClass(II.OpInfo[Op].RegClass);
}

/// CriticalPathStep - Return the next SUnit after SU on the bottom-up
/// critical path.
static SDep *CriticalPathStep(SUnit *SU) {
  SDep *Next = 0;
  unsigned NextDepth = 0;
  // Find the predecessor edge with the greatest depth.
  for (SUnit::pred_iterator P = SU->Preds.begin(), PE = SU->Preds.end();
       P != PE; ++P) {
    SUnit *PredSU = P->getSUnit();
    unsigned PredLatency = P->getLatency();
    unsigned PredTotalLatency = PredSU->getDepth() + PredLatency;
    // In the case of a latency tie, prefer an anti-dependency edge over
    // other types of edges.
    if (NextDepth < PredTotalLatency ||
        (NextDepth == PredTotalLatency && P->getKind() == SDep::Anti)) {
      NextDepth = PredTotalLatency;
      Next = &*P;
    }
  }
  return Next;
}

/// BreakAntiDependencies - Identifiy anti-dependencies along the critical path
/// of the ScheduleDAG and break them by renaming registers.
///
bool SchedulePostRATDList::BreakAntiDependencies() {
  // The code below assumes that there is at least one instruction,
  // so just duck out immediately if the block is empty.
  if (SUnits.empty()) return false;

  // Find the node at the bottom of the critical path.
  SUnit *Max = 0;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    SUnit *SU = &SUnits[i];
    if (!Max || SU->getDepth() + SU->Latency > Max->getDepth() + Max->Latency)
      Max = SU;
  }

  DOUT << "Critical path has total latency "
       << (Max->getDepth() + Max->Latency) << "\n";

  // Track progress along the critical path through the SUnit graph as we walk
  // the instructions.
  SUnit *CriticalPathSU = Max;
  MachineInstr *CriticalPathMI = CriticalPathSU->getInstr();

  // For live regs that are only used in one register class in a live range,
  // the register class. If the register is not live, the corresponding value
  // is null. If the register is live but used in multiple register classes,
  // the corresponding value is -1 casted to a pointer.
  const TargetRegisterClass *
    Classes[TargetRegisterInfo::FirstVirtualRegister] = {};

  // Map registers to all their references within a live range.
  std::multimap<unsigned, MachineOperand *> RegRefs;

  // The index of the most recent kill (proceding bottom-up), or ~0u if
  // the register is not live.
  unsigned KillIndices[TargetRegisterInfo::FirstVirtualRegister];
  std::fill(KillIndices, array_endof(KillIndices), ~0u);
  // The index of the most recent complete def (proceding bottom up), or ~0u if
  // the register is live.
  unsigned DefIndices[TargetRegisterInfo::FirstVirtualRegister];
  std::fill(DefIndices, array_endof(DefIndices), BB->size());

  // Determine the live-out physregs for this block.
  if (BB->back().getDesc().isReturn())
    // In a return block, examine the function live-out regs.
    for (MachineRegisterInfo::liveout_iterator I = MRI.liveout_begin(),
         E = MRI.liveout_end(); I != E; ++I) {
      unsigned Reg = *I;
      Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
      KillIndices[Reg] = BB->size();
      DefIndices[Reg] = ~0u;
      // Repeat, for all aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        unsigned AliasReg = *Alias;
        Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
        KillIndices[AliasReg] = BB->size();
        DefIndices[AliasReg] = ~0u;
      }
    }
  else
    // In a non-return block, examine the live-in regs of all successors.
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         SE = BB->succ_end(); SI != SE; ++SI) 
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
           E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
        KillIndices[Reg] = BB->size();
        DefIndices[Reg] = ~0u;
        // Repeat, for all aliases.
        for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
          unsigned AliasReg = *Alias;
          Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
          KillIndices[AliasReg] = BB->size();
          DefIndices[AliasReg] = ~0u;
        }
      }

  // Consider callee-saved registers as live-out, since we're running after
  // prologue/epilogue insertion so there's no way to add additional
  // saved registers.
  //
  // TODO: If the callee saves and restores these, then we can potentially
  // use them between the save and the restore. To do that, we could scan
  // the exit blocks to see which of these registers are defined.
  // Alternatively, callee-saved registers that aren't saved and restored
  // could be marked live-in in every block.
  for (const unsigned *I = TRI->getCalleeSavedRegs(); *I; ++I) {
    unsigned Reg = *I;
    Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
    KillIndices[Reg] = BB->size();
    DefIndices[Reg] = ~0u;
    // Repeat, for all aliases.
    for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
      unsigned AliasReg = *Alias;
      Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
      KillIndices[AliasReg] = BB->size();
      DefIndices[AliasReg] = ~0u;
    }
  }

  // Consider this pattern:
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  //   A = ...
  //   ... = A
  // There are three anti-dependencies here, and without special care,
  // we'd break all of them using the same register:
  //   A = ...
  //   ... = A
  //   B = ...
  //   ... = B
  //   B = ...
  //   ... = B
  //   B = ...
  //   ... = B
  // because at each anti-dependence, B is the first register that
  // isn't A which is free.  This re-introduces anti-dependencies
  // at all but one of the original anti-dependencies that we were
  // trying to break.  To avoid this, keep track of the most recent
  // register that each register was replaced with, avoid avoid
  // using it to repair an anti-dependence on the same register.
  // This lets us produce this:
  //   A = ...
  //   ... = A
  //   B = ...
  //   ... = B
  //   C = ...
  //   ... = C
  //   B = ...
  //   ... = B
  // This still has an anti-dependence on B, but at least it isn't on the
  // original critical path.
  //
  // TODO: If we tracked more than one register here, we could potentially
  // fix that remaining critical edge too. This is a little more involved,
  // because unlike the most recent register, less recent registers should
  // still be considered, though only if no other registers are available.
  unsigned LastNewReg[TargetRegisterInfo::FirstVirtualRegister] = {};

  // Attempt to break anti-dependence edges on the critical path. Walk the
  // instructions from the bottom up, tracking information about liveness
  // as we go to help determine which registers are available.
  bool Changed = false;
  unsigned Count = BB->size() - 1;
  for (MachineBasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend();
       I != E; ++I, --Count) {
    MachineInstr *MI = &*I;

    // After regalloc, IMPLICIT_DEF instructions aren't safe to treat as
    // dependence-breaking. In the case of an INSERT_SUBREG, the IMPLICIT_DEF
    // is left behind appearing to clobber the super-register, while the
    // subregister needs to remain live. So we just ignore them.
    if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF)
      continue;

    // Check if this instruction has a dependence on the critical path that
    // is an anti-dependence that we may be able to break. If it is, set
    // AntiDepReg to the non-zero register associated with the anti-dependence.
    //
    // We limit our attention to the critical path as a heuristic to avoid
    // breaking anti-dependence edges that aren't going to significantly
    // impact the overall schedule. There are a limited number of registers
    // and we want to save them for the important edges.
    // 
    // TODO: Instructions with multiple defs could have multiple
    // anti-dependencies. The current code here only knows how to break one
    // edge per instruction. Note that we'd have to be able to break all of
    // the anti-dependencies in an instruction in order to be effective.
    unsigned AntiDepReg = 0;
    if (MI == CriticalPathMI) {
      if (SDep *Edge = CriticalPathStep(CriticalPathSU)) {
        SUnit *NextSU = Edge->getSUnit();

        // Only consider anti-dependence edges.
        if (Edge->getKind() == SDep::Anti) {
          AntiDepReg = Edge->getReg();
          assert(AntiDepReg != 0 && "Anti-dependence on reg0?");
          // Don't break anti-dependencies on non-allocatable registers.
          if (!AllocatableSet.test(AntiDepReg))
            AntiDepReg = 0;
          else {
            // If the SUnit has other dependencies on the SUnit that it
            // anti-depends on, don't bother breaking the anti-dependency
            // since those edges would prevent such units from being
            // scheduled past each other regardless.
            //
            // Also, if there are dependencies on other SUnits with the
            // same register as the anti-dependency, don't attempt to
            // break it.
            for (SUnit::pred_iterator P = CriticalPathSU->Preds.begin(),
                 PE = CriticalPathSU->Preds.end(); P != PE; ++P)
              if (P->getSUnit() == NextSU ?
                    (P->getKind() != SDep::Anti || P->getReg() != AntiDepReg) :
                    (P->getKind() == SDep::Data && P->getReg() == AntiDepReg)) {
                AntiDepReg = 0;
                break;
              }
          }
        }
        CriticalPathSU = NextSU;
        CriticalPathMI = CriticalPathSU->getInstr();
      } else {
        // We've reached the end of the critical path.
        CriticalPathSU = 0;
        CriticalPathMI = 0;
      }
    }

    // Scan the register operands for this instruction and update
    // Classes and RegRefs.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      const TargetRegisterClass *NewRC =
        getInstrOperandRegClass(TRI, TII, MI->getDesc(), i);

      // If this instruction has a use of AntiDepReg, breaking it
      // is invalid.
      if (MO.isUse() && AntiDepReg == Reg)
        AntiDepReg = 0;

      // For now, only allow the register to be changed if its register
      // class is consistent across all uses.
      if (!Classes[Reg] && NewRC)
        Classes[Reg] = NewRC;
      else if (!NewRC || Classes[Reg] != NewRC)
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);

      // Now check for aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        // If an alias of the reg is used during the live range, give up.
        // Note that this allows us to skip checking if AntiDepReg
        // overlaps with any of the aliases, among other things.
        unsigned AliasReg = *Alias;
        if (Classes[AliasReg]) {
          Classes[AliasReg] = reinterpret_cast<TargetRegisterClass *>(-1);
          Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);
        }
      }

      // If we're still willing to consider this register, note the reference.
      if (Classes[Reg] != reinterpret_cast<TargetRegisterClass *>(-1))
        RegRefs.insert(std::make_pair(Reg, &MO));
    }

    // Determine AntiDepReg's register class, if it is live and is
    // consistently used within a single class.
    const TargetRegisterClass *RC = AntiDepReg != 0 ? Classes[AntiDepReg] : 0;
    assert((AntiDepReg == 0 || RC != NULL) &&
           "Register should be live if it's causing an anti-dependence!");
    if (RC == reinterpret_cast<TargetRegisterClass *>(-1))
      AntiDepReg = 0;

    // Look for a suitable register to use to break the anti-depenence.
    //
    // TODO: Instead of picking the first free register, consider which might
    // be the best.
    if (AntiDepReg != 0) {
      for (TargetRegisterClass::iterator R = RC->allocation_order_begin(MF),
           RE = RC->allocation_order_end(MF); R != RE; ++R) {
        unsigned NewReg = *R;
        // Don't replace a register with itself.
        if (NewReg == AntiDepReg) continue;
        // Don't replace a register with one that was recently used to repair
        // an anti-dependence with this AntiDepReg, because that would
        // re-introduce that anti-dependence.
        if (NewReg == LastNewReg[AntiDepReg]) continue;
        // If NewReg is dead and NewReg's most recent def is not before
        // AntiDepReg's kill, it's safe to replace AntiDepReg with NewReg.
        assert(((KillIndices[AntiDepReg] == ~0u) != (DefIndices[AntiDepReg] == ~0u)) &&
               "Kill and Def maps aren't consistent for AntiDepReg!");
        assert(((KillIndices[NewReg] == ~0u) != (DefIndices[NewReg] == ~0u)) &&
               "Kill and Def maps aren't consistent for NewReg!");
        if (KillIndices[NewReg] == ~0u &&
            Classes[NewReg] != reinterpret_cast<TargetRegisterClass *>(-1) &&
            KillIndices[AntiDepReg] <= DefIndices[NewReg]) {
          DOUT << "Breaking anti-dependence edge on "
               << TRI->getName(AntiDepReg)
               << " with " << RegRefs.count(AntiDepReg) << " references"
               << " using " << TRI->getName(NewReg) << "!\n";

          // Update the references to the old register to refer to the new
          // register.
          std::pair<std::multimap<unsigned, MachineOperand *>::iterator,
                    std::multimap<unsigned, MachineOperand *>::iterator>
             Range = RegRefs.equal_range(AntiDepReg);
          for (std::multimap<unsigned, MachineOperand *>::iterator
               Q = Range.first, QE = Range.second; Q != QE; ++Q)
            Q->second->setReg(NewReg);

          // We just went back in time and modified history; the
          // liveness information for the anti-depenence reg is now
          // inconsistent. Set the state as if it were dead.
          Classes[NewReg] = Classes[AntiDepReg];
          DefIndices[NewReg] = DefIndices[AntiDepReg];
          KillIndices[NewReg] = KillIndices[AntiDepReg];

          Classes[AntiDepReg] = 0;
          DefIndices[AntiDepReg] = KillIndices[AntiDepReg];
          KillIndices[AntiDepReg] = ~0u;

          RegRefs.erase(AntiDepReg);
          Changed = true;
          LastNewReg[AntiDepReg] = NewReg;
          break;
        }
      }
    }

    // Update liveness.
    // Proceding upwards, registers that are defed but not used in this
    // instruction are now dead.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      if (!MO.isDef()) continue;
      // Ignore two-addr defs.
      if (MI->isRegReDefinedByTwoAddr(i)) continue;

      DefIndices[Reg] = Count;
      KillIndices[Reg] = ~0u;
      Classes[Reg] = 0;
      RegRefs.erase(Reg);
      // Repeat, for all subregs.
      for (const unsigned *Subreg = TRI->getSubRegisters(Reg);
           *Subreg; ++Subreg) {
        unsigned SubregReg = *Subreg;
        DefIndices[SubregReg] = Count;
        KillIndices[SubregReg] = ~0u;
        Classes[SubregReg] = 0;
        RegRefs.erase(SubregReg);
      }
      // Conservatively mark super-registers as unusable.
      for (const unsigned *Super = TRI->getSuperRegisters(Reg);
           *Super; ++Super) {
        unsigned SuperReg = *Super;
        Classes[SuperReg] = reinterpret_cast<TargetRegisterClass *>(-1);
      }
    }
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      if (!MO.isUse()) continue;

      const TargetRegisterClass *NewRC =
        getInstrOperandRegClass(TRI, TII, MI->getDesc(), i);

      // For now, only allow the register to be changed if its register
      // class is consistent across all uses.
      if (!Classes[Reg] && NewRC)
        Classes[Reg] = NewRC;
      else if (!NewRC || Classes[Reg] != NewRC)
        Classes[Reg] = reinterpret_cast<TargetRegisterClass *>(-1);

      RegRefs.insert(std::make_pair(Reg, &MO));

      // It wasn't previously live but now it is, this is a kill.
      if (KillIndices[Reg] == ~0u) {
        KillIndices[Reg] = Count;
        DefIndices[Reg] = ~0u;
      }
      // Repeat, for all aliases.
      for (const unsigned *Alias = TRI->getAliasSet(Reg); *Alias; ++Alias) {
        unsigned AliasReg = *Alias;
        if (KillIndices[AliasReg] == ~0u) {
          KillIndices[AliasReg] = Count;
          DefIndices[AliasReg] = ~0u;
        }
      }
    }
  }
  assert(Count == ~0u && "Count mismatch!");

  return Changed;
}

//===----------------------------------------------------------------------===//
//  Top-Down Scheduling
//===----------------------------------------------------------------------===//

/// ReleaseSucc - Decrement the NumPredsLeft count of a successor. Add it to
/// the PendingQueue if the count reaches zero. Also update its cycle bound.
void SchedulePostRATDList::ReleaseSucc(SUnit *SU, SDep *SuccEdge) {
  SUnit *SuccSU = SuccEdge->getSUnit();
  --SuccSU->NumPredsLeft;
  
#ifndef NDEBUG
  if (SuccSU->NumPredsLeft < 0) {
    cerr << "*** Scheduling failed! ***\n";
    SuccSU->dump(this);
    cerr << " has been released too many times!\n";
    assert(0);
  }
#endif
  
  // Compute how many cycles it will be before this actually becomes
  // available.  This is the max of the start time of all predecessors plus
  // their latencies.
  SuccSU->setDepthToAtLeast(SU->getDepth() + SuccEdge->getLatency());
  
  if (SuccSU->NumPredsLeft == 0) {
    PendingQueue.push_back(SuccSU);
  }
}

/// ScheduleNodeTopDown - Add the node to the schedule. Decrement the pending
/// count of its successors. If a successor pending count is zero, add it to
/// the Available queue.
void SchedulePostRATDList::ScheduleNodeTopDown(SUnit *SU, unsigned CurCycle) {
  DOUT << "*** Scheduling [" << CurCycle << "]: ";
  DEBUG(SU->dump(this));
  
  Sequence.push_back(SU);
  assert(CurCycle >= SU->getDepth() && "Node scheduled above its depth!");
  SU->setDepthToAtLeast(CurCycle);

  // Top down: release successors.
  for (SUnit::succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
       I != E; ++I)
    ReleaseSucc(SU, &*I);

  SU->isScheduled = true;
  AvailableQueue.ScheduledNode(SU);
}

/// ListScheduleTopDown - The main loop of list scheduling for top-down
/// schedulers.
void SchedulePostRATDList::ListScheduleTopDown() {
  unsigned CurCycle = 0;

  // All leaves to Available queue.
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    // It is available if it has no predecessors.
    if (SUnits[i].Preds.empty()) {
      AvailableQueue.push(&SUnits[i]);
      SUnits[i].isAvailable = true;
    }
  }
  
  // While Available queue is not empty, grab the node with the highest
  // priority. If it is not ready put it back.  Schedule the node.
  std::vector<SUnit*> NotReady;
  Sequence.reserve(SUnits.size());
  while (!AvailableQueue.empty() || !PendingQueue.empty()) {
    // Check to see if any of the pending instructions are ready to issue.  If
    // so, add them to the available queue.
    unsigned MinDepth = ~0u;
    for (unsigned i = 0, e = PendingQueue.size(); i != e; ++i) {
      if (PendingQueue[i]->getDepth() <= CurCycle) {
        AvailableQueue.push(PendingQueue[i]);
        PendingQueue[i]->isAvailable = true;
        PendingQueue[i] = PendingQueue.back();
        PendingQueue.pop_back();
        --i; --e;
      } else if (PendingQueue[i]->getDepth() < MinDepth)
        MinDepth = PendingQueue[i]->getDepth();
    }
    
    // If there are no instructions available, don't try to issue anything, and
    // don't advance the hazard recognizer.
    if (AvailableQueue.empty()) {
      CurCycle = MinDepth != ~0u ? MinDepth : CurCycle + 1;
      continue;
    }

    SUnit *FoundSUnit = 0;

    bool HasNoopHazards = false;
    while (!AvailableQueue.empty()) {
      SUnit *CurSUnit = AvailableQueue.pop();

      ScheduleHazardRecognizer::HazardType HT =
        HazardRec->getHazardType(CurSUnit);
      if (HT == ScheduleHazardRecognizer::NoHazard) {
        FoundSUnit = CurSUnit;
        break;
      }

      // Remember if this is a noop hazard.
      HasNoopHazards |= HT == ScheduleHazardRecognizer::NoopHazard;

      NotReady.push_back(CurSUnit);
    }

    // Add the nodes that aren't ready back onto the available list.
    if (!NotReady.empty()) {
      AvailableQueue.push_all(NotReady);
      NotReady.clear();
    }

    // If we found a node to schedule, do it now.
    if (FoundSUnit) {
      ScheduleNodeTopDown(FoundSUnit, CurCycle);
      HazardRec->EmitInstruction(FoundSUnit);

      // If this is a pseudo-op node, we don't want to increment the current
      // cycle.
      if (FoundSUnit->Latency)  // Don't increment CurCycle for pseudo-ops!
        ++CurCycle;
    } else if (!HasNoopHazards) {
      // Otherwise, we have a pipeline stall, but no other problem, just advance
      // the current cycle and try again.
      DOUT << "*** Advancing cycle, no work to do\n";
      HazardRec->AdvanceCycle();
      ++NumStalls;
      ++CurCycle;
    } else {
      // Otherwise, we have no instructions to issue and we have instructions
      // that will fault if we don't do this right.  This is the case for
      // processors without pipeline interlocks and other cases.
      DOUT << "*** Emitting noop\n";
      HazardRec->EmitNoop();
      Sequence.push_back(0);   // NULL here means noop
      ++NumNoops;
      ++CurCycle;
    }
  }

#ifndef NDEBUG
  VerifySchedule(/*isBottomUp=*/false);
#endif
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createPostRAScheduler() {
  return new PostRAScheduler();
}
