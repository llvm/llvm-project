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

  using RegisterSet = SetVector<Register>;

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

  void dump() {
    for (auto SI : RegisterMap) {
      dbgs() << "\nMBB: " << SI.first;
      dbgs() << "\n\tW: ";
      for (auto R : SI.second.ActiveSet) {
        dbgs() << printReg(R) << " ";
      }
      dbgs() << "\n\tS: ";
      for (auto R : SI.second.SpillSet) {
        dbgs() << printReg(R) << " ";
      }
      dbgs() << "\n";
    }
  }

  void init(MachineFunction &MF, bool IsVGPRs) {
    IsVGPRsPass = IsVGPRs;
    ST = &MF.getSubtarget<GCNSubtarget>();
    MRI = &MF.getRegInfo();
    MFI = &MF.getFrameInfo();
    TRI = ST->getRegisterInfo();
    TII = ST->getInstrInfo();
    NumAvailableRegs =
        IsVGPRsPass ? ST->getMaxNumVGPRs(MF) : ST->getMaxNumSGPRs(MF);
    //  ? TRI->getRegPressureSetLimit(
    //        MF, AMDGPU::RegisterPressureSets::VGPR_32)
    //  : TRI->getRegPressureSetLimit(
    //        MF, AMDGPU::RegisterPressureSets::SReg_32);
  }

  SpillInfo &getBlockInfo(const MachineBasicBlock &MBB);

  void processFunction(MachineFunction &MF);
  void processBlock(MachineBasicBlock &MBB);
  void processLoop(MachineLoop *L);
  void connectToPredecessors(MachineBasicBlock &MBB, bool IgnoreLoops = false);
  void initActiveSetUsualBlock(MachineBasicBlock &MBB);
  void initActiveSetLoopHeader(MachineBasicBlock &MBB);

  void reloadAtEnd(MachineBasicBlock &MBB, Register VReg);
  void spillAtEnd(MachineBasicBlock &MBB, Register VReg);
  void reloadBefore(Register VReg, MachineBasicBlock::iterator InsertBefore);
  void spillBefore(Register VReg, MachineBasicBlock::iterator InsertBefore);

  unsigned getLoopMaxRP(MachineLoop *L);
  void limit(MachineBasicBlock &MBB, RegisterSet &Active, RegisterSet &Spilled,
             MachineBasicBlock::iterator I,
             const RegisterSet Defs = RegisterSet());

  unsigned getSizeInRegs(const Register VReg);
  unsigned getSizeInRegs(const RegisterSet VRegs);
  bool takeReg(Register R) {
    return ((IsVGPRsPass && TRI->isVGPR(*MRI, R)) ||
            (!IsVGPRsPass && TRI->isSGPRReg(*MRI, R)));
  }

  unsigned fillActiveSet(MachineBasicBlock &MBB, RegisterSet S,
                         unsigned Capacity = 0);

public:
  AMDGPUSSASpiller() = default;

  AMDGPUSSASpiller(const LiveIntervals &LIS, MachineLoopInfo &LI,
                   MachineDominatorTree &MDT, AMDGPUNextUseAnalysis::Result &NU)
      : LIS(LIS), LI(LI), MDT(MDT), NU(NU) {}
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
  for (auto MBB : RPOT) {
    if (LI.isLoopHeader(MBB)) {
      initActiveSetLoopHeader(*MBB);
    } else {
      initActiveSetUsualBlock(*MBB);
    }
    connectToPredecessors(*MBB);
    processBlock(*MBB);
    dump();
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
}

void AMDGPUSSASpiller::processBlock(MachineBasicBlock &MBB) {
  auto &Entry = RegisterMap[MBB.getNumber()];
  RegisterSet &Active = Entry.ActiveSet;
  RegisterSet &Spilled = Entry.SpillSet;
  
  for (MachineBasicBlock::iterator I : MBB) {
    RegisterSet Reloads;
    for (auto U : I->uses()) {
      if (!U.isReg())
        continue;
      if (U.getReg().isPhysical())
        continue;
      Register VReg = U.getReg();
      if (!takeReg(VReg))
        continue;
      // if (U.getSubReg()) {
      //   dbgs() << U << "\n";
      // }
      if (Active.insert(VReg)) {
        // Not in reg, hence, should have been spilled before
        // FIXME: This is ODD as the Spilled set is a union among all
        // predecessors and should already contain all spilled before!
        // SPECIAL CASE: undef
        if (!U.isUndef()) {
          Spilled.insert(VReg);
          Reloads.insert(VReg);
        }
      }
    }
    RegisterSet Defs;
    for (auto D : I->defs()) {
      if (D.getReg().isVirtual() && takeReg(D.getReg()))
        Defs.insert(D.getReg());
    }

    if (Reloads.empty() && Defs.empty())
      continue;

    limit(MBB, Active, Spilled, I);
    limit(MBB, Active, Spilled, I, Defs);
    // FIXME: limit with Defs is assumed to create room for the registers being
    // defined by I. Calling with std::next(I) makes spills inserted AFTER I!!!
    Active.insert(Defs.begin(), Defs.end());
    // Add reloads for VRegs in Reloads before I
    for (auto R : Reloads)
      reloadBefore(R, I);
  }
  // Now, clear dead registers.
  RegisterSet Deads;
  for (auto R : Active) {
    if (NU.isDead(MBB, R))
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
    dumpRegSet(getBlockInfo(*Pred).SpillSet);
    Entry.SpillSet.set_union(getBlockInfo(*Pred).SpillSet);
    dumpRegSet(Entry.SpillSet);
  }
  set_intersect(Entry.SpillSet, Entry.ActiveSet);
  for (auto Pred : Preds) {
    auto PE = getBlockInfo(*Pred);
    RegisterSet ReloadInPred = set_difference(Entry.ActiveSet, PE.ActiveSet);
    if (!ReloadInPred.empty()) {
      // We're about to insert N reloads at the end of the predecessor block.
      // Make sure we have enough registers for N definitions or spill to make
      // room for them.
      limit(*Pred, PE.ActiveSet, PE.SpillSet, Pred->end(), ReloadInPred);
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
      LiveIn.insert(VReg);
    }
  }

  for (auto &PHI : MBB.phis()) {
    for (auto U : PHI.uses()) {
      if (U.isReg() && takeReg(U.getReg())) {
        // assume PHIs operands are always virtual regs
        LiveIn.insert(U.getReg());
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

void AMDGPUSSASpiller::reloadAtEnd(MachineBasicBlock &MBB, Register VReg) {
  reloadBefore(VReg, MBB.getFirstInstrTerminator());
}

void AMDGPUSSASpiller::spillAtEnd(MachineBasicBlock &MBB, Register VReg) {
  spillBefore(VReg, MBB.getFirstTerminator());
}

void AMDGPUSSASpiller::reloadBefore(Register VReg,
                                    MachineBasicBlock::iterator InsertBefore) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
  int FI = MFI->CreateSpillStackObject(TRI->getRegSizeInBits(*RC),
                                       TRI->getSpillAlign(*RC));
  TII->loadRegFromStackSlot(*InsertBefore->getParent(), InsertBefore, VReg, FI,
                            RC, TRI, VReg);
}

void AMDGPUSSASpiller::spillBefore(Register VReg,
                                   MachineBasicBlock::iterator InsertBefore) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
  int FI = MFI->CreateSpillStackObject(TRI->getRegSizeInBits(*RC),
                                       TRI->getSpillAlign(*RC));
  TII->storeRegToStackSlot(*InsertBefore->getParent(), InsertBefore, VReg, true, FI,
                            RC, TRI, VReg);
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
                             MachineBasicBlock::iterator I,
                             const RegisterSet Defs) {
  MachineBasicBlock::iterator LimitPoint = I;

  if (!Defs.empty()) {
    if (LimitPoint != MBB.end())
      LimitPoint = std::next(LimitPoint);
  }

  unsigned CurRP = getSizeInRegs(Active);
  unsigned DefsRP = getSizeInRegs(Defs);
  if(CurRP <= NumAvailableRegs - DefsRP)
    return;

  if (LimitPoint == MBB.end())
    NU.getSortedForBlockEnd(MBB, Active);
  else
    NU.getSortedForInstruction(*LimitPoint, Active);


  unsigned ShrinkTo = NumAvailableRegs - DefsRP;
  RegisterSet ToSpill;
  while (CurRP > ShrinkTo) {
    auto R = Active.pop_back_val();
    unsigned RegSize = getSizeInRegs(R);
    CurRP -= RegSize;
    if (!Spilled.contains(R))
      ToSpill.insert(R);
  }

  for (auto R : ToSpill) {

    bool Alive = (LimitPoint == MBB.end()) ? !NU.isDead(MBB, R)
                                           : !NU.isDead(*LimitPoint, R);
    if (Alive) {
      spillBefore(R, I);
      Spilled.insert(R);
    }
  }
}

unsigned AMDGPUSSASpiller::getSizeInRegs(const Register VReg) {
  const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
  return TRI->getRegClassWeight(RC).RegWeight;
}

unsigned AMDGPUSSASpiller::getSizeInRegs(const RegisterSet VRegs) {
  unsigned Size = 0;
  for (auto VReg : VRegs) {
    Size += getSizeInRegs(VReg);
  }
  return Size;
}

unsigned AMDGPUSSASpiller::fillActiveSet(MachineBasicBlock &MBB, RegisterSet S,
                                         unsigned Capacity) {
  unsigned Limit = Capacity ? Capacity : NumAvailableRegs;
  auto &Active = RegisterMap[MBB.getNumber()].ActiveSet;
  unsigned Size = getSizeInRegs(Active);
  NU.getSortedForInstruction(*MBB.instr_begin(), S);
  for (auto VReg : S) {
    unsigned RSize = getSizeInRegs(VReg);
    if (Size + RSize < Limit) {
      Active.insert(VReg);
      Size += RSize;
    }
  }
  return Size;
}

bool AMDGPUSSASpiller::run(MachineFunction &MF) {
  init(MF, false);
  processFunction(MF);
  init(MF, true);
  processFunction(MF);
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
