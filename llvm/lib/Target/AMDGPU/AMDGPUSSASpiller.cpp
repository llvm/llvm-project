#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Timer.h"
#include "llvm/Target/TargetMachine.h"

#include "AMDGPUNextUseAnalysis.h"
#include "GCNRegPressure.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-spiller"

namespace {

  static void dumpRegSet(SetVector<Register> VRegs) {
    dbgs() << "\n";
    for (auto R : VRegs) {
      dbgs() << printReg(R) << " ";
    }
    dbgs() << "\n";
  }

class AMDGPUSSASpiller : public PassInfoMixin <AMDGPUSSASpiller> {
  const LiveIntervals &LIS;
  MachineLoopInfo &LI;
  MachineDominatorTree &MDT;
  AMDGPUNextUseAnalysis::Result &NU;
  const MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  const GCNSubtarget *ST;
  MachineFrameInfo *MFI;

  static constexpr int NO_STACK_SLOT = INT_MAX;

  unsigned NumSpillSlots;

  DenseMap<VRegMaskPair, unsigned> Virt2StackSlotMap;

  // TODO: HOW TO MAP VREG + LANEMASK TO SPILL SLOT ???

  // IF IT EVEN POSSIBLE TO SPILL REG.SUBREG ?

  // CREATE NEW PSEUDOS SI_SPILL_XXX_SAVE/RESTORE_WITH_SUBREG ???

  unsigned createSpillSlot(const TargetRegisterClass *RC) {
    unsigned Size = TRI->getSpillSize(*RC);
    Align Alignment = TRI->getSpillAlign(*RC);
    // TODO: See VirtRegMap::createSpillSlot - if we need to bother with
    // TRI->canRealignStack(*MF) ?
    int SS = MFI->CreateSpillStackObject(Size, Alignment);
    ++NumSpillSlots;
    return SS;
  }

  unsigned assignVirt2StackSlot(VRegMaskPair VMP) {
    assert(VMP.VReg.isVirtual());
    assert(!Virt2StackSlotMap.contains(VMP) &&
           "attempt to assign stack slot to already spilled register");
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
  
    TG = new TimerGroup("SSA SPiller Timing", "Time Spent in different parts of the SSA Spiller");
    T1 = new Timer("General time", "ProcessFunction", *TG);
    T2 = new Timer("Limit", "Time spent in limit()", *TG);
    T3 = new Timer("Initialization time", "Init Active Sets", *TG);
    T4 = new Timer("Instruction processing time", "Process Instruction w/o limit", *TG);

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

  void reloadAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP);
  void spillAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP);
  void reloadBefore(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP);
  void spillBefore(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP);

  unsigned getLoopMaxRP(MachineLoop *L);
  void limit(MachineBasicBlock &MBB, RegisterSet &Active, RegisterSet &Spilled,
             MachineBasicBlock::iterator I, unsigned Limit,
             RegisterSet &ToSpill);

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
  }

  unsigned fillActiveSet(MachineBasicBlock &MBB, RegisterSet S,
                         unsigned Capacity = 0);

public:
  AMDGPUSSASpiller() = default;

  AMDGPUSSASpiller(const LiveIntervals &LIS, MachineLoopInfo &LI,
                   MachineDominatorTree &MDT, AMDGPUNextUseAnalysis::Result &NU)
      : LIS(LIS), LI(LI), MDT(MDT), NU(NU),
        NumSpillSlots(0), Virt2StackSlotMap(NO_STACK_SLOT) {}
  ~AMDGPUSSASpiller() {
    delete TG;
    delete T2;
    delete T3;
    delete T4;
    // delete TG;
      }
  bool run(MachineFunction &MF);
};

AMDGPUSSASpiller::SpillInfo &
AMDGPUSSASpiller::getBlockInfo(const MachineBasicBlock &MBB) {
  if (!RegisterMap.contains(MBB.getNumber()))
    llvm::report_fatal_error("Incorrect MF walk order");
  return RegisterMap[MBB.getNumber()];
}

void AMDGPUSSASpiller::processFunction(MachineFunction &MF) {
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  
  T1->startTimer();
  for (auto MBB : RPOT) {
    
    T3->startTimer();
    if (LI.isLoopHeader(MBB)) {
      initActiveSetLoopHeader(*MBB);
    } else {
      initActiveSetUsualBlock(*MBB);
    }
    connectToPredecessors(*MBB);
    T3->stopTimer();
    processBlock(*MBB);
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
  T1->stopTimer();
}

void AMDGPUSSASpiller::processBlock(MachineBasicBlock &MBB) {
  auto &Entry = RegisterMap[MBB.getNumber()];
  RegisterSet &Active = Entry.ActiveSet;
  RegisterSet &Spilled = Entry.SpillSet;
  
  for (MachineBasicBlock::iterator I : MBB) {
    RegisterSet Reloads;
    T4->startTimer();
    for (auto U : I->uses()) {
      if (!U.isReg())
        continue;
      if (U.getReg().isPhysical())
        continue;
      Register VReg = U.getReg();
      if (!takeReg(VReg))
        continue;
      
      VRegMaskPair VMP(U, *TRI);
      
      if (Active.insert(VMP)) {
        // Not in reg, hence, should have been spilled before
        // FIXME: This is ODD as the Spilled set is a union among all
        // predecessors and should already contain all spilled before!
        // SPECIAL CASE: undef
        if (!U.isUndef()) {
          Spilled.insert(VMP);
          Reloads.insert(VMP);
        }
      }
    }
    RegisterSet Defs;
    for (auto D : I->defs()) {
      if (D.getReg().isVirtual() && takeReg(D.getReg()))
        Defs.insert(VRegMaskPair(D, *TRI));
    }

    if (Reloads.empty() && Defs.empty()) {
      T4->stopTimer();
      continue;
    }
    T4->stopTimer();

    RegisterSet ToSpill;
    limit(MBB, Active, Spilled, I, NumAvailableRegs, ToSpill);
    limit(MBB, Active, Spilled, std::next(I),
          NumAvailableRegs - getSizeInRegs(Defs), ToSpill);
    T4->startTimer();
    for (auto R : ToSpill) {
      spillBefore(MBB, I, R);
      Spilled.insert(R);
    }
    // FIXME: limit with Defs is assumed to create room for the registers being
    // defined by I. Calling with std::next(I) makes spills inserted AFTER I!!!
    Active.insert(Defs.begin(), Defs.end());
    // Add reloads for VRegs in Reloads before I
    for (auto R : Reloads)
      reloadBefore(MBB, I, R);
    T4->stopTimer();
  }
  // Now, clear dead registers.
  RegisterSet Deads;
  for (auto R : Active) {
    if (NU.isDead(MBB, MBB.end(), R))
      Deads.insert(R);
  }
  Active.set_subtract(Deads);
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

  auto &Entry = RegisterMap[MBB.getNumber()];
  SmallVector<MachineBasicBlock *> Preds(predecessors(&MBB));

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
    //dumpRegSet(getBlockInfo(*Pred).SpillSet);
    Entry.SpillSet.set_union(getBlockInfo(*Pred).SpillSet);
    //dumpRegSet(Entry.SpillSet);
  }
  set_intersect(Entry.SpillSet, Entry.ActiveSet);
  for (auto Pred : Preds) {
    auto PE = getBlockInfo(*Pred);
    RegisterSet ReloadInPred = set_difference(Entry.ActiveSet, PE.ActiveSet);
    if (!ReloadInPred.empty()) {
      // We're about to insert N reloads at the end of the predecessor block.
      // Make sure we have enough registers for N definitions or spill to make
      // room for them.
      RegisterSet ToSpill;
      limit(*Pred, PE.ActiveSet, PE.SpillSet, Pred->end(),
            NumAvailableRegs - getSizeInRegs(ReloadInPred), ToSpill);
      for (auto R : ToSpill) {
        spillBefore(*Pred, Pred->end(), R);
        PE.SpillSet.insert(R);
      }
      for (auto R : ReloadInPred) {
        reloadAtEnd(*Pred, R);
        // FIXME: Do we need to update sets?
        PE.ActiveSet.insert(R);
      }
    }

    for (auto S : set_intersection(set_difference(Entry.SpillSet, PE.SpillSet),
                                   PE.ActiveSet)) {
      spillAtEnd(*Pred, S);
      // FIXME: Do we need to update sets?
      PE.SpillSet.insert(S);
      Entry.SpillSet.insert(S);
    }
  }
}

void AMDGPUSSASpiller::initActiveSetUsualBlock(MachineBasicBlock &MBB) {

  if (predecessors(&MBB).empty())
    return;

  auto Pred = MBB.pred_begin();

  RegisterSet Take = getBlockInfo(**Pred).ActiveSet;
  RegisterSet Cand = getBlockInfo(**Pred).ActiveSet;

  for (Pred = std::next(Pred); Pred != MBB.pred_end(); ++Pred) {
    set_intersect(Take, getBlockInfo(**Pred).ActiveSet);
    Cand.set_union(getBlockInfo(**Pred).ActiveSet);
  }
  Cand.set_subtract(Take);

  if (Take.empty() && Cand.empty())
    return;

  unsigned TakeSize = fillActiveSet(MBB, Take);
  if (TakeSize < NumAvailableRegs) {
    unsigned FullSize = fillActiveSet(MBB, Cand);
    assert(FullSize <= NumAvailableRegs);
  }
}

void AMDGPUSSASpiller::initActiveSetLoopHeader(MachineBasicBlock &MBB) {
  // auto &Entry = RegisterMap[MBB.getNumber()];
  RegisterSet LiveIn;

  for (unsigned i = 0; i < MRI->getNumVirtRegs(); i++) {
    Register VReg = Register::index2VirtReg(i);
    if (!LIS.hasInterval(VReg))
      continue;
    if (takeReg(VReg) && LIS.isLiveInToMBB(LIS.getInterval(VReg), &MBB)) {
      LiveIn.insert({VReg, LaneBitmask::getAll()});
    }
  }

  for (auto &PHI : MBB.phis()) {
    for (auto U : PHI.uses()) {
      if (U.isReg() && takeReg(U.getReg())) {
        // assume PHIs operands are always virtual regs
        LiveIn.insert(VRegMaskPair(U, *TRI));
      }
    }
  }

  RegisterSet UsedInLoop;
  MachineLoop *L = LI.getLoopFor(&MBB);
  for (auto B : L->blocks()) {
    RegisterSet Tmp(NU.usedInBlock(*B));
    UsedInLoop.set_union(Tmp);
  }

  // Take - LiveIns used in Loop. Cand - LiveThrough
  RegisterSet Take = set_intersection(LiveIn, UsedInLoop);
  RegisterSet Cand = set_difference(LiveIn, UsedInLoop);


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
}

const TargetRegisterClass *
AMDGPUSSASpiller::getRegClassForVregMaskPair(VRegMaskPair VMP,
                                             unsigned &SubRegIdx) {
  const TargetRegisterClass *RC;

  if (VMP.LaneMask.all()) {
    RC = TRI->getRegClassForReg(*MRI, VMP.VReg);
  } else {
    SmallVector<unsigned> Idxs;
    if (TRI->getCoveringSubRegIndexes(*MRI, RC, VMP.LaneMask, Idxs)) {
      SubRegIdx = Idxs[0];
      for (int i = 1; i < Idxs.size() - 1; i++)
        SubRegIdx = TRI->composeSubRegIndices(SubRegIdx, Idxs[i]);
      RC = TRI->getSubRegisterClass(RC, SubRegIdx);
    }
  }

  return RC;
}

void AMDGPUSSASpiller::reloadAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP) {
  reloadBefore(MBB, MBB.getFirstInstrTerminator(), VMP);
}

void AMDGPUSSASpiller::spillAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP) {
  spillBefore(MBB, MBB.getFirstTerminator(), VMP);
}

void AMDGPUSSASpiller::reloadBefore(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator InsertBefore,
                                    VRegMaskPair VMP) {
  unsigned SubRegIdx = 0;
  const TargetRegisterClass *RC = getRegClassForVregMaskPair(VMP, SubRegIdx);
  int FI = getStackSlot(VMP);
  TII->loadRegFromStackSlot(MBB, InsertBefore, VMP.VReg, FI,
                            RC, TRI, VMP.VReg, SubRegIdx);
}

void AMDGPUSSASpiller::spillBefore(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator InsertBefore,
                                   VRegMaskPair VMP) {
  unsigned SubRegIdx = 0;
  const TargetRegisterClass *RC = getRegClassForVregMaskPair(VMP, SubRegIdx);

  int FI = assignVirt2StackSlot(VMP);
  TII->storeRegToStackSlot(MBB, InsertBefore, VMP.VReg, true, FI, RC, TRI,
                           VMP.VReg, SubRegIdx);
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

void AMDGPUSSASpiller::limit(MachineBasicBlock &MBB, RegisterSet &Active,
                             RegisterSet &Spilled,
                             MachineBasicBlock::iterator I, unsigned Limit,
                             RegisterSet &ToSpill) {

  T2->startTimer();
  Active.remove_if([&](VRegMaskPair P) { return NU.isDead(MBB, I, P); });

  unsigned CurRP = getSizeInRegs(Active);
  if (CurRP <= Limit) {
    T2->stopTimer();
    return;
  }

  sortRegSetAt(MBB, I, Active);

  // Here we expect that the furthest use is use of %4:VReg_1024.sub_31 so its
  // size is 32bits

  while (CurRP > Limit) {
    auto P = Active.pop_back_val();
    unsigned RegSize = getSizeInRegs(P.VReg);
    CurRP -= RegSize;
    if (!Spilled.contains(P))
      ToSpill.insert(P);
  }
  T2->stopTimer();
}

unsigned AMDGPUSSASpiller::getSizeInRegs(const VRegMaskPair VMP) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
  return TRI->getRegClassWeight(RC).RegWeight;
}

unsigned AMDGPUSSASpiller::getSizeInRegs(const RegisterSet VRegs) {
  unsigned Size = 0;
  for (auto VMP : VRegs) {
    Size += getSizeInRegs(VMP.VReg);
  }
  return Size;
}

unsigned AMDGPUSSASpiller::fillActiveSet(MachineBasicBlock &MBB, RegisterSet S,
                                         unsigned Capacity) {
  unsigned Limit = Capacity ? Capacity : NumAvailableRegs;
  auto &Active = RegisterMap[MBB.getNumber()].ActiveSet;
  unsigned Size = getSizeInRegs(Active);
  sortRegSetAt(MBB, MBB.begin(), S);
  for (auto VMP : S) {
    unsigned RSize = getSizeInRegs(VMP.VReg);
    if (Size + RSize < Limit) {
      Active.insert(VMP);
      Size += RSize;
    }
  }
  return Size;
}

bool AMDGPUSSASpiller::run(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  MRI = &MF.getRegInfo();
  MFI = &MF.getFrameInfo();
  TRI = ST->getRegisterInfo();
  TII = ST->getInstrInfo();

  Virt2StackSlotMap.resize(MRI->getNumVirtRegs());

  init(MF, false);
  processFunction(MF);
  init(MF, true);

  processFunction(MF);
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
  const LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
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

llvm::PassPluginLibraryInfo getMyNewMachineFunctionPassPluginInfo() {
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
  return getMyNewMachineFunctionPassPluginInfo();
}
