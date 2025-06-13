#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Timer.h"
#include "llvm/Target/TargetMachine.h"

#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPUSSARAUtils.h"
#include "GCNRegPressure.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-spiller"

namespace {

  

class AMDGPUSSASpiller : public PassInfoMixin <AMDGPUSSASpiller> {
  LiveIntervals &LIS;
  MachineLoopInfo &LI;
  MachineDominatorTree &MDT;
  AMDGPUNextUseAnalysis::Result &NU;
  MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  const GCNSubtarget *ST;
  MachineFrameInfo *MFI;

  static constexpr int NO_STACK_SLOT = INT_MAX;

  unsigned NumSpillSlots;

  DenseMap<VRegMaskPair, unsigned> Virt2StackSlotMap;
  DenseMap<VRegMaskPair, MachineInstr *> SpillPoints;
  DenseSet<unsigned> ProcessedBlocks;

  LLVM_ATTRIBUTE_NOINLINE void dumpRegSet(SetVector<VRegMaskPair> VMPs);

  unsigned createSpillSlot(const TargetRegisterClass *RC) {
    unsigned Size = TRI->getSpillSize(*RC);
    Align Alignment = TRI->getSpillAlign(*RC);
    // TODO: See VirtRegMap::createSpillSlot - if we need to bother with
    // TRI->canRealignStack(*MF) ?
    int SS = MFI->CreateSpillStackObject(Size, Alignment);
    ++NumSpillSlots;
    return SS;
  }

  // return existing stack slot if any or assigns the new one
  unsigned assignVirt2StackSlot(VRegMaskPair VMP) {
    assert(VMP.VReg.isVirtual());
    if (Virt2StackSlotMap.contains(VMP))
      return Virt2StackSlotMap[VMP];
    const TargetRegisterClass *RC = MRI->getRegClass(VMP.VReg);
    return Virt2StackSlotMap[VMP] = createSpillSlot(RC);
  }

  unsigned getStackSlot(VRegMaskPair VMP) {
    assert(VMP.VReg.isVirtual());
    return Virt2StackSlotMap[VMP];
  }

  TimerGroup *TG;
  Timer *T1;
  Timer *T2;
  Timer *T3;
  Timer *T4;

  using RegisterSet = SetVector<VRegMaskPair>;

  struct SpillInfo {
    //MachineBasicBlock *Parent;
    RegisterSet ActiveSet;
    RegisterSet SpillSet;
  };

  bool IsVGPRsPass;
  unsigned NumAvailableRegs;
  DenseMap<unsigned, SpillInfo> RegisterMap;
  DenseMap<unsigned, unsigned> PostponedLoopLatches;
  DenseMap<unsigned, SmallVector<unsigned>> LoopHeader2Latches;

  void printVRegMaskPair(const VRegMaskPair P) {
    SmallVector<unsigned> Idxs;
    const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, P.VReg);
    bool HasSubReg = TRI->getCoveringSubRegIndexes(*MRI, RC, P.LaneMask, Idxs);
    dbgs() << "Vreg: ";
    if (HasSubReg)
      for (auto i : Idxs)
        dbgs() << printReg(P.VReg, TRI, i, MRI) << "]\n";
    else
      dbgs() << printReg(P.VReg) << "]\n";
  }

  void dump() {
    for (auto SI : RegisterMap) {
      dbgs() << "\nMBB: " << SI.first;
      dbgs() << "\n\tW: ";
      for (auto P : SI.second.ActiveSet) {
        printVRegMaskPair(P);
      }
      dbgs() << "\n\tS: ";
      for (auto P : SI.second.SpillSet) {
        printVRegMaskPair(P);
      }
      dbgs() << "\n";
    }
  }

  void init(MachineFunction &MF, bool IsVGPRs) {
    IsVGPRsPass = IsVGPRs;
  
  

    NumAvailableRegs =
        IsVGPRsPass ? ST->getMaxNumVGPRs(MF) : ST->getMaxNumSGPRs(MF);
    //  ? TRI->getRegPressureSetLimit(
    //        MF, AMDGPU::RegisterPressureSets::VGPR_32)
    //  : TRI->getRegPressureSetLimit(
    //        MF, AMDGPU::RegisterPressureSets::SReg_32);
    RegisterMap.clear();
  }

  SpillInfo &getBlockInfo(const MachineBasicBlock &MBB);

  void processFunction(MachineFunction &MF);
  void processBlock(MachineBasicBlock &MBB);
  void processLoop(MachineLoop *L);
  void connectToPredecessors(MachineBasicBlock &MBB, bool IgnoreLoops = false);
  void initActiveSetUsualBlock(MachineBasicBlock &MBB);
  void initActiveSetLoopHeader(MachineBasicBlock &MBB);

  Register reloadAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP);
  void spillAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP);
  Register reloadBefore(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP);
  void spillBefore(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP);

  void rewriteUses(MachineBasicBlock &MBB, Register OldVReg, Register NewVReg);

  unsigned getLoopMaxRP(MachineLoop *L);
  // Returns number of spilled VRegs
  unsigned limit(MachineBasicBlock &MBB, RegisterSet &Active, RegisterSet &Spilled,
             MachineBasicBlock::iterator I, unsigned Limit);

  unsigned getSizeInRegs(const VRegMaskPair VMP);
  unsigned getSizeInRegs(const RegisterSet VRegs);

  const TargetRegisterClass *getRegClassForVregMaskPair(VRegMaskPair VMP,
                                                        unsigned &SubRegIdx);

  bool takeReg(Register R) {
    return ((IsVGPRsPass && TRI->isVGPR(*MRI, R)) ||
            (!IsVGPRsPass && TRI->isSGPRReg(*MRI, R)));
  }

  void sortRegSetAt(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    RegisterSet &VRegs) {
    DenseMap<VRegMaskPair, unsigned> M;
    bool BlockEnd = I == MBB.end();
    for (auto VMP : VRegs)
      M[VMP] = BlockEnd ? NU.getNextUseDistance(MBB, VMP)
                      : NU.getNextUseDistance(I, VMP);

    auto SortByDist = [&](const VRegMaskPair LHS, const VRegMaskPair RHS) {
      return M[LHS] < M[RHS];
    };

    SmallVector<VRegMaskPair> Tmp(VRegs.takeVector());
    sort(Tmp, SortByDist);
    VRegs.insert(Tmp.begin(), Tmp.end());
    LLVM_DEBUG(dbgs() << "\nActive set sorted at " << *I;
    for (auto P : VRegs) {
      printVRegMaskPair(P);
      dbgs() << " : " << M[P] << "\n";
    });
  }

  // Fills Active until reaches the NumAvailableRegs. If @Capacity is passed
  // fills exactly this number of regs.
  unsigned fillActiveSet(MachineBasicBlock &MBB, RegisterSet S,
                         unsigned Capacity = 0);

  bool isCoveredActive(VRegMaskPair VMP, const RegisterSet Active);

public:
  AMDGPUSSASpiller() = default;

  AMDGPUSSASpiller(LiveIntervals &LIS, MachineLoopInfo &LI,
                   MachineDominatorTree &MDT, AMDGPUNextUseAnalysis::Result &NU)
      : LIS(LIS), LI(LI), MDT(MDT), NU(NU), NumSpillSlots(0) //,
        // Virt2StackSlotMap(NO_STACK_SLOT) {
        {
    TG = new TimerGroup("SSA SPiller Timing",
                        "Time Spent in different parts of the SSA Spiller");
    T1 = new Timer("General time", "ProcessFunction", *TG);
    T2 = new Timer("Limit", "Time spent in limit()", *TG);
    T3 = new Timer("Initialization time", "Init Active Sets", *TG);
    T4 = new Timer("Instruction processing time",
                   "Process Instruction w/o limit", *TG);
  }
  ~AMDGPUSSASpiller() {
    delete TG;
    delete T2;
    delete T3;
    delete T4;
    //delete TG;
      }
  bool run(MachineFunction &MF);
};

LLVM_ATTRIBUTE_NOINLINE void
AMDGPUSSASpiller::dumpRegSet(SetVector<VRegMaskPair> VMPs) {
  dbgs() << "\n";
  for (auto P : VMPs) {
    printVRegMaskPair(P);
    dbgs() << "\n";
  }
  dbgs() << "\n";
}

LLVM_ATTRIBUTE_NOINLINE void
AMDGPUSSASpiller::printVRegMaskPair(const VRegMaskPair P) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, P.VReg);
  LaneBitmask FullMask = getFullMaskForRC(*RC, TRI);
  dbgs() << "Vreg: [";
  if (P.LaneMask == FullMask) {
    dbgs() << printReg(P.VReg) << "] ";
  } else {
    unsigned SubRegIndex = getSubRegIndexForLaneMask(P.LaneMask, TRI);
    dbgs() << printReg(P.VReg, TRI, SubRegIndex, MRI) << "] ";
  }
}

AMDGPUSSASpiller::SpillInfo &
AMDGPUSSASpiller::getBlockInfo(const MachineBasicBlock &MBB) {
  if (!RegisterMap.contains(MBB.getNumber()))
    llvm::report_fatal_error("Incorrect MF walk order");
  return RegisterMap[MBB.getNumber()];
}

void AMDGPUSSASpiller::processFunction(MachineFunction &MF) {
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  
  // T1->startTimer();
  for (auto MBB : RPOT) {
    
    // T3->startTimer();
    if (LI.isLoopHeader(MBB)) {
      initActiveSetLoopHeader(*MBB);
    } else {
      initActiveSetUsualBlock(*MBB);
    }
    connectToPredecessors(*MBB);
    // T3->stopTimer();
    processBlock(*MBB);
    ProcessedBlocks.insert(MBB->getNumber());
    // dump();
    // We process loop blocks twice: once with Spill/Active sets of
    // loop latch blocks unknown, and then again as soon as the latch blocks
    // sets are computed.
    if (PostponedLoopLatches.contains(MBB->getNumber())) {
      SmallVector<unsigned> &Latches =
          LoopHeader2Latches[PostponedLoopLatches[MBB->getNumber()]];
      remove_if(Latches, [MBB](int Num) { return Num == MBB->getNumber(); });
      if (Latches.empty()) {
        processLoop(LI.getLoopFor(MBB));
      }
      PostponedLoopLatches.erase(MBB->getNumber());
    }
  }
  ProcessedBlocks.clear();
  // T1->stopTimer();
}

void AMDGPUSSASpiller::processBlock(MachineBasicBlock &MBB) {
  auto &Entry = RegisterMap[MBB.getNumber()];
  RegisterSet &Active = Entry.ActiveSet;
  RegisterSet &Spilled = Entry.SpillSet;
  
  // for (MachineBasicBlock::iterator I : MBB) {
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); I++) {
    RegisterSet Reloads;
    // T4->startTimer();
    for (auto &U : I->uses()) {
      if (!U.isReg())
        continue;
      if (U.getReg().isPhysical())
        continue;
      Register VReg = U.getReg();
      if (!takeReg(VReg))
        continue;

      VRegMaskPair VMP(U, TRI, MRI);

      // We don't need to make room for the PHI uses as they operands must
      // already present in the corresponding predecessor Active set! Just
      // make sure they really are.
      if (I->isPHI()) {
        auto OpNo = U.getOperandNo();
        auto B = I->getOperand(++OpNo);
        assert(B.isMBB());
        MachineBasicBlock *ValueSrc = B.getMBB();
       
        if (ProcessedBlocks.contains(ValueSrc->getNumber())) {
          auto Info = getBlockInfo(*ValueSrc);
          dumpRegSet(Info.ActiveSet);
          assert(getBlockInfo(*ValueSrc).ActiveSet.contains(VMP) &&
                 "PHI node input value is not live out predecessor!");
        }
        continue;
      }

      if (!isCoveredActive(VMP, Active)) {
        // Not in reg, hence, should have been spilled before
        // FIXME: This is ODD as the Spilled set is a union among all
        // predecessors and should already contain all spilled before!
        // SPECIAL CASE: undef
        if (!U.isUndef()) {
          Reloads.insert(VMP);
        }
      }
    }

    if (I->isPHI()) {
      // We don't need to make room for the PHI-defined values as they will be
      // lowered to the copies at the end of the corresponding predecessors
      // and occupies the same register with the corresponding PHI input
      // value. Nevertheless, we must add them to the Active to indicate their
      // values are available.
      for (auto D : I->defs()) {
        Register R = D.getReg();
        if (takeReg(R)) {
          Active.insert(VRegMaskPair(D, TRI, MRI));
        }
      }
      continue;
    }

    RegisterSet Defs;
    for (auto D : I->defs()) {
      if (D.getReg().isVirtual() && takeReg(D.getReg()))
        Defs.insert(VRegMaskPair(D, TRI, MRI));
    }

    if (Reloads.empty() && Defs.empty()) {
      // T4->stopTimer();
      continue;
    }
    // T4->stopTimer();

    LLVM_DEBUG(dbgs() << "\nCurrent Active set is:\n"; dumpRegSet(Active));
    LLVM_DEBUG(dbgs() << "\nVRegs used but spilled before, we're to reload:\n";
               dumpRegSet(Reloads));

    Active.insert(Reloads.begin(), Reloads.end());
    Spilled.insert(Reloads.begin(), Reloads.end());

    LLVM_DEBUG(dbgs() << "\nActive set with uses reloaded:\n";
               dumpRegSet(Active));

    
    limit(MBB, Active, Spilled, I, NumAvailableRegs);
    unsigned NSpills = limit(MBB, Active, Spilled, std::next(I),
           NumAvailableRegs - getSizeInRegs(Defs));

    // T4->startTimer();


    Active.insert(Defs.begin(), Defs.end());
    // Add reloads for VRegs in Reloads before I
    for (auto R : Reloads) {
      LLVM_DEBUG(dbgs() << "\nReloading "; printVRegMaskPair(R);
                 dbgs() << "\n");
      Register NewVReg = reloadBefore(MBB, I, R);
      rewriteUses(MBB, R.VReg, NewVReg);
    }

    std::advance(I, NSpills);
    // T4->stopTimer();
  }
  // Now, clear dead registers. We generally take care of trimming deads at the
  // entry to "limit". The dangling deads may appear when operand is SGPR but
  // result is VGPR, so we don't enter to the limit second time to make room for
  // the result. If this is the last use of the SGPR operand it is effectively
  // dead.
  // %X:sreg_32 = ...
  //   ***
  // %Y:vgpr_32 = COPY %X:sreg_32 <-- %X is dead but we won't call "limit" for
  // %Y in this pass.
  RegisterSet Deads;
  for (auto R : Active) {
    if (NU.isDead(MBB, MBB.end(), R))
      Deads.insert(R);
  }

  if (!Deads.empty()) {
    LLVM_DEBUG(dbgs() << "\nThese VRegs are DEAD at the end of MBB_"
                      << MBB.getNumber() << "." << MBB.getName() << "\n";
               dumpRegSet(Deads));
    Active.set_subtract(Deads);
    LLVM_DEBUG(dbgs() << "\nActive set after DEAD VRegs removed:\n";
               dumpRegSet(Active));
  }

  // Take care of the LiveOuts which are Succ's PHI operands.
  for (auto Succ : successors(&MBB)) {
    for (auto &PHI : Succ->phis()) {
      for (auto &U : PHI.uses()) {
        if (U.isReg() && takeReg(U.getReg())) {
          auto OpNo = U.getOperandNo();
          auto B = PHI.getOperand(++OpNo);
          assert(B.isMBB());
          MachineBasicBlock *ValueSrc = B.getMBB();
          if (ValueSrc->getNumber() == MBB.getNumber()) {
            VRegMaskPair VMP(U, TRI, MRI);
            if (!isCoveredActive(VMP, Active)) {
              Register NewVReg = reloadAtEnd(MBB, VMP);
              // U.setReg(NewVReg);
              // U.setSubReg(AMDGPU::NoRegister);

              // The code below is commented out because of the BUG in
              // MachineSSAUpdater. In case the register class of a PHI operand
              // defined register is a superclass of a NewReg it inserts a COPY
              // AFTER the PHI

              //  Predecessor:
              //  %157:vgpr_32 = SI_SPILL_V32_RESTORE %stack.0

              //  %146:vreg_64 = PHI %70:vreg_64.sub0, %bb.3, %144:vgpr_32, %bb.1

              // becomes:

              // %146:vreg_64 = PHI %158:vreg_64.sub0, %bb.3, %144:vgpr_32,
              // %bb.1 %158:vreg_64 = COPY %157

              MachineSSAUpdater SSAUpddater(*MBB.getParent());
              SSAUpddater.Initialize(U.getReg());
              SSAUpddater.AddAvailableValue(&MBB, NewVReg);
              SSAUpddater.RewriteUse(U);
            }
          }
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "\nActive set after Succs PHI operands processing:\n";
             dumpRegSet(Active));
}

void AMDGPUSSASpiller::processLoop(MachineLoop *L) {
  for (auto MBB : L->getBlocks()) {
    connectToPredecessors(*MBB, true);
    processBlock(*MBB);
  }
}

void AMDGPUSSASpiller::connectToPredecessors(MachineBasicBlock &MBB,
                                             bool IgnoreLoops) {
  if (predecessors(&MBB).empty())
    return;

  LLVM_DEBUG(dbgs() << "\nconnectToPredecessors block " << MBB.getName());
  auto &Entry = RegisterMap[MBB.getNumber()];
  SmallVector<MachineBasicBlock *> Preds(predecessors(&MBB));

  RegisterSet PHIOps;
  for (auto &PHI : MBB.phis()) {
    for (auto &PU : PHI.uses()) {
      if (PU.isReg()) {
        if (takeReg(PU.getReg())) {
          VRegMaskPair P(PU, TRI, MRI);
          PHIOps.insert(P);
        }
      }
    }
  }

  // in RPOT loop latches have not been processed yet
  // their Active and Spill sets are not yet known
  // Exclude from processing and postpone.
  if (!IgnoreLoops && LI.isLoopHeader(&MBB)) {
    MachineLoop *L = LI.getLoopFor(&MBB);
    SmallVector<MachineBasicBlock *> Latches;
    L->getLoopLatches(Latches);
    for (auto LL : Latches) {
      remove_if(Preds, [LL](MachineBasicBlock *BB) {
        return LL->getNumber() == BB->getNumber();
      });
      LoopHeader2Latches[MBB.getNumber()].push_back(LL->getNumber());
      PostponedLoopLatches[LL->getNumber()] = MBB.getNumber();
    }
  }

  for (auto Pred : Preds) {
    dumpRegSet(getBlockInfo(*Pred).SpillSet);
    Entry.SpillSet.set_union(getBlockInfo(*Pred).SpillSet);
    dumpRegSet(Entry.SpillSet);
  }
  // The line below was added according to algorithm proposed in Hack&Broun.
  // It is commented out because of the following observation:
  // If some reister is spilled in block it is not in its active set anymore.
  // If this block has the only one successor, then the successor active set is
  // equal to the block active set. Then the line below removes the spilled
  // register from its spilled set and will not propagate it to the successors
  // along the CFG. If we have later on a join block with multiple predecessors,
  // then the spilled register will not be spilled along the path to that join
  // block from the common dominator.
  //              BB0 [x active]
  //              / \
  //           BB1   \                [x spilled]
  //            |    |
  //           BB2   |                [x is not in BB1 Active set =>
  //             \   |                it is not in BB2 Active set =>
  //              \  |              BB2.Spilled ^ BB2.Active yeilds empty set]
  //               \/
  //               BB3 [x is not in BB2 Spilled set => will not be spilled along
  //               the BB0 -> BB3 edge. If we have ause of x inBB3 reload will
  //               fail if the CF reached BB3 along the BB0 -> BB3 edge]

  // set_intersect(Entry.SpillSet, Entry.ActiveSet);

  for (auto Pred : Preds) {
    auto &PE = getBlockInfo(*Pred);
    LLVM_DEBUG(dbgs() << "\nCurr block [ MBB_" << MBB.getNumber() << "."
                      << MBB.getName() << " ] Active Set:\n";
               dumpRegSet(Entry.ActiveSet);
               dbgs() << "\nPred [ MBB_" << Pred->getNumber() << "."
                      << Pred->getName() << " ] ActiveSet:\n";
               dumpRegSet(PE.ActiveSet));
    RegisterSet Tmp = set_difference(Entry.ActiveSet, PE.ActiveSet);
    dumpRegSet(Tmp);
    // Pred LiveOuts which are current block PHI operands don't need to be
    // active across both edges.
    RegisterSet ReloadInPred = set_difference(Tmp, PHIOps);
    dumpRegSet(ReloadInPred);
    set_intersect(ReloadInPred, PE.SpillSet);
    dumpRegSet(ReloadInPred);
    if (!ReloadInPred.empty()) {

      // Since we operate on SSA, any register that is live across the edge must
      // either be defined before or within the IDom, or be a PHI operand. If a
      // register is neither a PHI operand nor live-out from all predecessors,
      // it must have been spilled in one of them. Registers that are defined
      // and used entirely within a predecessor are dead at its exit. Therefore,
      // there is always room to reload a register that is not live across the
      // edge.

      for (auto R : ReloadInPred) {
        Register NewVReg = reloadAtEnd(*Pred, R);
        rewriteUses(*Pred, R.VReg, NewVReg);
      }
    }

    LLVM_DEBUG(dbgs() << "\nPred [ MBB_" << Pred->getNumber() << "."
                      << Pred->getName() << " ] SpillSet:\n";
               dumpRegSet(PE.SpillSet));
    for (auto S : set_intersection(set_difference(Entry.SpillSet, PE.SpillSet),
                                   PE.ActiveSet)) {
      spillAtEnd(*Pred, S);
      PE.SpillSet.insert(S);
      PE.ActiveSet.remove(S);
      Entry.SpillSet.insert(S);
      Entry.ActiveSet.remove(S);
    }
  }
}

void AMDGPUSSASpiller::initActiveSetUsualBlock(MachineBasicBlock &MBB) {

  if (predecessors(&MBB).empty())
    return;

  LLVM_DEBUG(dbgs() << "Init Active Set " << MBB.getName() << "\n");
  auto Pred = MBB.pred_begin();

  RegisterSet Take = getBlockInfo(**Pred).ActiveSet;
  RegisterSet Cand = getBlockInfo(**Pred).ActiveSet;

  LLVM_DEBUG(dbgs() << "Pred's " << (*Pred)->getNumber() << " ActiveSet :";
             dumpRegSet(Take));

  for (Pred = std::next(Pred); Pred != MBB.pred_end(); ++Pred) {
    LLVM_DEBUG(auto PredsActive = getBlockInfo(**Pred).ActiveSet;
               dbgs() << "Pred's " << (*Pred)->getNumber() << " ActiveSet :";
               dumpRegSet(PredsActive));
    set_intersect(Take, getBlockInfo(**Pred).ActiveSet);
    Cand.set_union(getBlockInfo(**Pred).ActiveSet);
  }
  Cand.set_subtract(Take);

  if (Take.empty() && Cand.empty())
    return;

  LLVM_DEBUG(dbgs()<< "Take : "; dumpRegSet(Take));
  LLVM_DEBUG(dbgs()<< "Cand : "; dumpRegSet(Cand));

  unsigned TakeSize = fillActiveSet(MBB, Take);
  if (TakeSize < NumAvailableRegs) {
    unsigned FullSize = fillActiveSet(MBB, Cand);
    assert(FullSize <= NumAvailableRegs);
  }
  LLVM_DEBUG(dbgs() << MBB.getName() << "Exit ActiveSet: ";
             dumpRegSet(getBlockInfo(MBB).ActiveSet));
}

void AMDGPUSSASpiller::initActiveSetLoopHeader(MachineBasicBlock &MBB) {
  // auto &Entry = RegisterMap[MBB.getNumber()];
  RegisterSet LiveIn;

  for (unsigned i = 0; i < MRI->getNumVirtRegs(); i++) {
    Register VReg = Register::index2VirtReg(i);
    if (!LIS.hasInterval(VReg))
      continue;
  
    if (takeReg(VReg) && LIS.isLiveInToMBB(LIS.getInterval(VReg), &MBB)) {
      // we have to take care ofthe subreg index and set LaneMask accordingly
      // LaneBitmask LaneMask = LaneBitmask::getAll();
      // RegisterSet Preds;
      // for (auto Pred : MBB.predecessors()) {
      //   auto PredActive = getBlockInfo(*Pred).ActiveSet;
      //   set_intersect()
      //   for (auto P : PredActive) {
      //     if (P.VReg == VReg) {
      //       LaneMask = P.LaneMask;
      //       break;
      //     }
      //   }
      // }
      const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
      LiveIn.insert(VRegMaskPair(VReg, getFullMaskForRC(*RC, TRI)));
    }
  }

  LLVM_DEBUG(dbgs() << "\nBlock " << MBB.getName() << " Live Ins: ";
             dumpRegSet(LiveIn));

  // FIXME: We forced to collect pred's spill here so, maybe we need to move
  // pred's spill processing from connectToPredecessors to init? Or at least
  // don't do it again in connectToPredecessors if it is already done here?
  auto &Entry = RegisterMap[MBB.getNumber()];
  auto &Spilled = Entry.SpillSet;
  for (auto P : predecessors(&MBB)) {
     Spilled.set_union(getBlockInfo(*P).SpillSet);
  }

  RegisterSet UsedInLoop;
  MachineLoop *L = LI.getLoopFor(&MBB);
  for (auto B : L->blocks()) {
    RegisterSet Tmp = NU.usedInBlock(*B);
    Tmp.remove_if([&](VRegMaskPair P) { return !takeReg(P.VReg); });
    LLVM_DEBUG(dbgs() << "\nBlock " << B->getName()
                      << " is part of the loop. Used in block: ";
               dumpRegSet(Tmp));
    UsedInLoop.set_union(Tmp);
  }

  LLVM_DEBUG(dbgs() << "Total used in loop: "; dumpRegSet(UsedInLoop));

  // Take - LiveIns used in Loop. Cand - LiveThrough
  RegisterSet Take = set_intersection(LiveIn, UsedInLoop);
  RegisterSet Cand = set_difference(LiveIn, UsedInLoop);
  // We don't want to reload those not used in the loop which have been already
  // spilled.
  Cand.set_subtract(Spilled);

  LLVM_DEBUG(dbgs() << "\nBlock " << MBB.getName() << "sets\n";
             dbgs() << "Take : "; dumpRegSet(Take); dbgs() << "Cand : ";
             dumpRegSet(Cand));

  unsigned TakeSize = fillActiveSet(MBB, Take);
  if (TakeSize < NumAvailableRegs) {
    // At this point we have to decide not for the current block only but for
    // the whole loop. We use the following heuristic: given that the Cand
    // register set constitutes of those registers which are live-through the
    // loop, let's consider LoopMaxRP - CandSize to be the RP caused by those,
    // used inside the loop. According to this, we can keep NumAvailableRegs -
    // (LoopMaxRP - Cand.size()) in the loop header active set.
    unsigned LoopMaxRP = getLoopMaxRP(L);
    unsigned FreeSpace = NumAvailableRegs - (LoopMaxRP - Cand.size());
    unsigned FullSize = fillActiveSet(MBB, Cand, FreeSpace);
    assert(FullSize <= NumAvailableRegs);
  }
  LLVM_DEBUG(dbgs() << "\nFinal Loop header Active :";
             dumpRegSet(getBlockInfo(MBB).ActiveSet));
}

const TargetRegisterClass *
AMDGPUSSASpiller::getRegClassForVregMaskPair(VRegMaskPair VMP,
                                             unsigned &SubRegIdx) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VMP.VReg);
  LaneBitmask Mask = getFullMaskForRC(*RC, TRI);
  if (VMP.LaneMask != Mask) {
    unsigned SubRegIdx = getSubRegIndexForLaneMask(VMP.LaneMask, TRI);
    RC = TRI->getSubRegisterClass(RC, SubRegIdx);
  }

  // if (!VMP.LaneMask.all()) {
  //   SmallVector<unsigned> Idxs;
  //   if (TRI->getCoveringSubRegIndexes(*MRI, RC, VMP.LaneMask, Idxs)) {
  //     SubRegIdx = Idxs[0];
  //     // FIXME:  Idxs.size() - 1 ?
  //     for (unsigned i = 1; i < Idxs.size() - 1; i++)
  //       SubRegIdx = TRI->composeSubRegIndices(SubRegIdx, Idxs[i]);
  //     RC = TRI->getSubRegisterClass(RC, SubRegIdx);
  //   }
  // }

  return RC;
}

Register AMDGPUSSASpiller::reloadAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP) {
  return reloadBefore(MBB, MBB.getFirstInstrTerminator(), VMP);
}

void AMDGPUSSASpiller::spillAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP) {
  spillBefore(MBB, MBB.getFirstTerminator(), VMP);
}

Register AMDGPUSSASpiller::reloadBefore(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator InsertBefore,
                                    VRegMaskPair VMP) {
  unsigned SubRegIdx = 0;
  const TargetRegisterClass *RC = getRegClassForVregMaskPair(VMP, SubRegIdx);
  int FI = getStackSlot(VMP);
  Register NewVReg = MRI->createVirtualRegister(RC);
  TII->loadRegFromStackSlot(MBB, InsertBefore, NewVReg, FI, RC, TRI, NewVReg);
  // FIXME: dirty hack! To avoid further changing the TargetInstrInfo interface.
  MachineInstr &ReloadMI = *(--InsertBefore);
  LIS.InsertMachineInstrInMaps(ReloadMI);

  LIS.createAndComputeVirtRegInterval(NewVReg);
  auto &Entry = getBlockInfo(MBB);
  Entry.ActiveSet.insert({NewVReg, getFullMaskForRC(*RC, TRI)});
  return NewVReg;
}

void AMDGPUSSASpiller::spillBefore(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator InsertBefore,
                                   VRegMaskPair VMP) {
  unsigned SubRegIdx = getSubRegIndexForLaneMask(VMP.LaneMask, TRI);
  const TargetRegisterClass *RC = getRegClassForVregMaskPair(VMP, SubRegIdx);

  int FI = assignVirt2StackSlot(VMP);
  TII->storeRegToStackSlot(MBB, InsertBefore, VMP.VReg, true, FI, RC, TRI,
                           VMP.VReg, SubRegIdx);
  // FIXME: dirty hack! To avoid further changing the TargetInstrInfo interface.
  MachineInstr &Spill = *(--InsertBefore);
  LIS.InsertMachineInstrInMaps(Spill);

  if (LIS.hasInterval(VMP.VReg)) {
    
    LIS.removeInterval(VMP.VReg);

    // LiveInterval &LI = LIS.getInterval(VMP.VReg);
    // SlotIndex KillIdx = LIS.getInstructionIndex(Spill);
    // auto LR = LI.find(KillIdx);
    // if (LR != LI.end()) {
    //   SlotIndex Start = LR->start;
    //   SlotIndex End = LR->end;
    //   if (Start < KillIdx) {
    //     LI.removeSegment(KillIdx, End);
    //   }
    // }
  }
  SpillPoints[VMP] = &Spill;
}

void AMDGPUSSASpiller::rewriteUses(MachineBasicBlock &MBB, Register OldVReg,
                                   Register NewVReg) {
  MachineSSAUpdater SSAUpdater(*MBB.getParent());
  SSAUpdater.Initialize(OldVReg);
  SSAUpdater.AddAvailableValue(&MBB, NewVReg);
  for (MachineOperand &UseOp : MRI->use_operands(OldVReg)) {
    MachineInstr *UseMI = UseOp.getParent();
    MachineBasicBlock *UseMBB = UseMI->getParent();

    if (UseMBB->getNumber() == MBB.getNumber()) {
      UseOp.setReg(NewVReg);
      UseOp.setSubReg(AMDGPU::NoRegister);
    } else {
      // We skip rewriting if SSAUpdater already has a dominating def for
      // this block
      if (SSAUpdater.HasValueForBlock(UseMBB))
        continue;
      // This rewrites the use to a PHI result or correct value
      SSAUpdater.RewriteUse(UseOp);
    }
  }
}

unsigned AMDGPUSSASpiller::getLoopMaxRP(MachineLoop *L) {
  unsigned MaxRP = 0;
  for (auto MBB : L->getBlocks()) {
    SlotIndex MBBEndSlot = LIS.getSlotIndexes()->getMBBEndIdx(MBB);
    GCNUpwardRPTracker RPT(LIS);
    RPT.reset(*MRI, MBBEndSlot);
    for (auto &MI : reverse(*MBB))
      RPT.recede(MI);
    GCNRegPressure RP = RPT.getMaxPressure();
    unsigned CurMaxRP =
        IsVGPRsPass ? RP.getVGPRNum(ST->hasGFX90AInsts()) : RP.getSGPRNum();
    if (CurMaxRP > MaxRP)
      MaxRP = CurMaxRP;
  }
  return MaxRP;
}

unsigned AMDGPUSSASpiller::limit(MachineBasicBlock &MBB, RegisterSet &Active,
                                 RegisterSet &Spilled,
                                 MachineBasicBlock::iterator I,
                                 unsigned Limit) {

  // T2->startTimer();
  unsigned NumSpills = 0;

  LLVM_DEBUG(dbgs() << "\nIn \"limit\" with Limit = " << Limit << "\n");

  Active.remove_if([&](VRegMaskPair P) { return NU.isDead(MBB, I, P); });

  LLVM_DEBUG(dbgs() << "\n\"limit\": Active set after DEAD VRegs removed:\n";
             dumpRegSet(Active));

  unsigned CurRP = getSizeInRegs(Active);
  if (CurRP <= Limit) {
    // T2->stopTimer();
    return NumSpills;
  }

  
  sortRegSetAt(MBB, I, Active);

  RegisterSet ToSpill;

  while (CurRP > Limit) {
    auto P = Active.pop_back_val();
    unsigned RegSize = getSizeInRegs(P);
    unsigned SizeToSpill = CurRP - Limit;
    if (RegSize > SizeToSpill) {

      LaneBitmask ActiveMask = P.LaneMask;

      SmallVector<VRegMaskPair> Sorted = NU.getSortedSubregUses(I, P);

      for (auto S : Sorted) {
        unsigned Size = getSizeInRegs(S);
        CurRP -= Size;
        if (!Spilled.contains(S))
          ToSpill.insert(S);
        ActiveMask &= (~S.LaneMask);
        if (CurRP == Limit)
          break;
      }

      if (ActiveMask.any()) {
        // Insert the remaining part of the P to the Active set.
        VRegMaskPair Q(P.VReg, ActiveMask);
        // printVRegMaskPair(Q);
        Active.insert(Q);
      }

    } else {
      CurRP -= RegSize;
      if (!Spilled.contains(P))
        ToSpill.insert(P);
    }
  }
  LLVM_DEBUG(dbgs() << "\nActive set after at the end of the \"limit\":\n";
             dumpRegSet(Active));
  for (auto R : ToSpill) {
    LLVM_DEBUG(dbgs() << "\nSpilling "; printVRegMaskPair(R));
    spillBefore(MBB, I, R);
    NumSpills++;
    Spilled.insert(R);
  }

  if (!ToSpill.empty()) {
    dbgs() << "\nActive set after spilling:\n";
      dumpRegSet(Active);
    dbgs() << "\nSpilled set after spilling:\n";
    dumpRegSet(Spilled);
  }

  LLVM_DEBUG(if (!ToSpill.empty()) {
    dbgs() << "\nActive set after spilling:\n";
    dumpRegSet(Active);
    dbgs() << "\nSpilled set after spilling:\n";
    dumpRegSet(Spilled);
  });
  // T2->stopTimer();
  return NumSpills;
}

unsigned AMDGPUSSASpiller::getSizeInRegs(const VRegMaskPair VMP) {
  unsigned SubRegIdx = 0;
  const TargetRegisterClass *RC = getRegClassForVregMaskPair(VMP, SubRegIdx);
  return TRI->getRegClassWeight(RC).RegWeight;
}

unsigned AMDGPUSSASpiller::getSizeInRegs(const RegisterSet VRegs) {
  unsigned Size = 0;
  for (auto VMP : VRegs) {
    Size += getSizeInRegs(VMP);
  }
  return Size;
}

unsigned AMDGPUSSASpiller::fillActiveSet(MachineBasicBlock &MBB, RegisterSet S,
                                         unsigned Capacity) {
  unsigned Limit = Capacity ? Capacity : NumAvailableRegs;
  auto &Active = RegisterMap[MBB.getNumber()].ActiveSet;
  unsigned Size = Capacity ? 0 : getSizeInRegs(Active);
  sortRegSetAt(MBB, MBB.getFirstNonPHI(), S);
  for (auto VMP : S) {
    unsigned RSize = getSizeInRegs(VMP);
    if (Size + RSize <= Limit) {
      Active.insert(VMP);
      Size += RSize;
    }
  }
  return Size;
}

bool AMDGPUSSASpiller::isCoveredActive(VRegMaskPair VMP,
                                       const RegisterSet Active) {
  // printVRegMaskPair(VMP);
  // dumpRegSet(Active);
  for (auto P : Active) {
    if (P.VReg == VMP.VReg) {
      return (P.LaneMask & VMP.LaneMask) == VMP.LaneMask;
    }
  }
  return false;
}

bool AMDGPUSSASpiller::run(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  MRI = &MF.getRegInfo();
  MFI = &MF.getFrameInfo();
  TRI = ST->getRegisterInfo();
  TII = ST->getInstrInfo();
  T1->startTimer();
  init(MF, false);
  processFunction(MF);
  init(MF, true);

  processFunction(MF);
  MF.viewCFG();
  T1->stopTimer();
  TG->print(llvm::errs());
  return false;
}
} // namespace

PreservedAnalyses
llvm::AMDGPUSSASpillerPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  MachineLoopInfo &LI = MFAM.getResult<MachineLoopAnalysis>(MF);
  MachineDominatorTree &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  AMDGPUNextUseAnalysis::Result &NU = MFAM.getResult<AMDGPUNextUseAnalysis>(MF);
  AMDGPUSSASpiller Impl(LIS, LI, MDT, NU);
  bool Changed = Impl.run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  // TODO: We could detect CFG changed.
  auto PA = getMachineFunctionPassPreservedAnalyses();
  return PA;
}

class AMDGPUSSASpillerLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUSSASpillerLegacy() : MachineFunctionPass(ID) {
    initializeAMDGPUSSASpillerLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU SSA Spiller"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequiredTransitiveID(MachineLoopInfoID);
    AU.addPreservedID(MachineLoopInfoID);
    AU.addRequiredTransitiveID(MachineDominatorsID);
    AU.addPreservedID(MachineDominatorsID);
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<AMDGPUNextUseAnalysisWrapper>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

bool AMDGPUSSASpillerLegacy::runOnMachineFunction(MachineFunction &MF) {
  LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MachineLoopInfo &LI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  AMDGPUNextUseAnalysis::Result &NU =
      getAnalysis<AMDGPUNextUseAnalysisWrapper>().getNU();
  AMDGPUSSASpiller Impl(LIS, LI, MDT, NU);
  return Impl.run(MF);
}

INITIALIZE_PASS_BEGIN(AMDGPUSSASpillerLegacy, DEBUG_TYPE, "AMDGPU SSA Spiller",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AMDGPUNextUseAnalysisWrapper)
INITIALIZE_PASS_END(AMDGPUSSASpillerLegacy, DEBUG_TYPE, "AMDGPU SSA Spiller",
                    false, false)

char AMDGPUSSASpillerLegacy::ID = 0;

char &llvm::AMDGPUSSASpillerLegacyID = AMDGPUSSASpillerLegacy::ID;

FunctionPass *llvm::createAMDGPUSSASpillerLegacyPass() {
  return new AMDGPUSSASpillerLegacy();
}

llvm::PassPluginLibraryInfo getAMDGPUSSASpillerPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AMDGPUSSASpiller",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, MachineFunctionPassManager &MFPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "amdgpu-ssa-spiller") {
                    MFPM.addPass(AMDGPUSSASpillerPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// Expose the pass to LLVM’s pass manager infrastructure
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAMDGPUSSASpillerPassPluginInfo();
}
