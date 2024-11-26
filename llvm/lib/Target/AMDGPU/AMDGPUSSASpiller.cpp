#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"

#include "AMDGPUNextUseAnalysis.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-spiller"

namespace {

class AMDGPUSSASpiller : public PassInfoMixin <AMDGPUSSASpiller> {
  LiveVariables &LV;
  MachineLoopInfo &LI;
  MachineDominatorTree &MDT;
  AMDGPUNextUseAnalysis::Result &NU;
  const MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;

  using RegisterSet = SetVector<Register>;

  struct SpillInfo {
    //MachineBasicBlock *Parent;
    RegisterSet ActiveSet;
    RegisterSet SpillSet;
  };

  unsigned NumAvailableSGPRs;
  unsigned NumAvailableVGPRs;
  unsigned NumAvailableAGPRs;
  DenseMap<unsigned, SpillInfo> RegisterMap;
  DenseMap<unsigned, unsigned> PostponedLoopLatches;
  DenseMap<unsigned, SmallVector<unsigned>> LoopHeader2Latches;

  void init(const MachineFunction &MF) {
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    MRI = &MF.getRegInfo();
    TRI = ST.getRegisterInfo();
    NumAvailableVGPRs = ST.getTotalNumVGPRs();
    NumAvailableSGPRs = ST.getTotalNumSGPRs();

    // FIXME: what is real num AGPRs available?

    NumAvailableAGPRs = NumAvailableVGPRs;
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
  void reloadBefore(Register, MachineBasicBlock::iterator InsertBefore);
  void spillBefore(Register, MachineBasicBlock::iterator InsertBefore);

  SmallVector<unsigned> getLoopMaxRP(MachineLoop *L);
  void limit(RegisterSet &Active, RegisterSet &Spilled,
             MachineBasicBlock::iterator I,
             const RegisterSet Defs = RegisterSet());
  void limit(RegisterSet &Active, RegisterSet &Spilled,
             MachineBasicBlock::iterator LimitPoint,
             MachineBasicBlock::iterator InsertionPoint,
             RegisterSet RegClassSubset, unsigned Limit);
  void splitByRegPressureSet(const RegisterSet In, RegisterSet &SGPRS,
                             unsigned &SGPRRP, RegisterSet &VGPRS,
                             unsigned &VGPRRP, RegisterSet &AGPRS,
                             unsigned &AGPRRP);
  void formActiveSet(const MachineBasicBlock &MBB, const RegisterSet Take,
                     const RegisterSet Cand, MachineLoop *L = nullptr);

public:
  AMDGPUSSASpiller() = default;

  AMDGPUSSASpiller(LiveVariables &LV, MachineLoopInfo &LI,
                   MachineDominatorTree &MDT, AMDGPUNextUseAnalysis::Result &NU)
      : LV(LV), LI(LI), MDT(MDT), NU(NU) {}
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
  auto &Entry = getBlockInfo(MBB);
  RegisterSet &Active = Entry.ActiveSet;
  RegisterSet &Spilled = Entry.SpillSet;
  RegisterSet Reloads;
  for (auto &I : MBB) {
    for (auto U : I.uses()) {
      if (!U.isReg())
        continue;
      if (U.getReg().isPhysical())
        continue;
      Register VReg = U.getReg();
      if (Active.insert(VReg)) {
        // Not in reg, hence, should have been spilled before
        // TODO: This is ODD as the Spilled set is a union among all
        // predecessors and should already contain all spilled before!
        // Spilled.insert(U.getReg());
        Reloads.insert(VReg);
      }
    }
    RegisterSet Defs;
    for (auto D : I.defs()) {
      if (D.getReg().isVirtual())
        Defs.insert(D.getReg());
    }
    limit(Active, Spilled, I);
    limit(Active, Spilled, std::next(&I), Defs);
    // FIXME: limit with Defs is assumed to create room for the registers being
    // defined by I. Calling with std::next(I) makes spills inserted AFTER I!!!
    Active.insert(Defs.begin(), Defs.end());
    // Add reloads for VRegs in Reloads before I
    for (auto R : Reloads)
      reloadBefore(R, I);
  }
}

void AMDGPUSSASpiller::processLoop(MachineLoop *L) {
  for (auto MBB : L->getBlocks()) {
    connectToPredecessors(*MBB, true);
    processBlock(*MBB);
  }
}

void AMDGPUSSASpiller::connectToPredecessors(MachineBasicBlock &MBB,
                                             bool IgnoreLoops) {

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

  SpillInfo &Cur = getBlockInfo(MBB);
  for (auto Pred : Preds) {
    Cur.SpillSet.set_union(getBlockInfo(*Pred).SpillSet);
  }
  set_intersect(Cur.SpillSet, Cur.ActiveSet);
  for (auto Pred : Preds) {
    for (auto R : set_difference(Cur.ActiveSet, getBlockInfo(*Pred).ActiveSet))
      reloadAtEnd(*Pred, R);

    for (auto S : set_intersection(
             set_difference(Cur.SpillSet, getBlockInfo(*Pred).SpillSet),
             getBlockInfo(*Pred).ActiveSet))
      spillAtEnd(*Pred, S);
  }
}

void AMDGPUSSASpiller::initActiveSetUsualBlock(MachineBasicBlock &MBB) {

  if (predecessors(&MBB).empty())
    return;

  auto Pred = MBB.pred_begin();

  RegisterSet Take = getBlockInfo(**Pred).ActiveSet;
  RegisterSet Cand = getBlockInfo(**Pred).ActiveSet;

  for (std::next(Pred); Pred != MBB.pred_end(); ++Pred) {
    set_intersect(Take, getBlockInfo(**Pred).ActiveSet);
    Cand.set_union(getBlockInfo(**Pred).ActiveSet);
  }
  Cand.set_subtract(Take);

  formActiveSet(MBB, Take, Cand);
}

void AMDGPUSSASpiller::initActiveSetLoopHeader(MachineBasicBlock &MBB) {
  auto &Entry = getBlockInfo(MBB);
  RegisterSet LiveIn;

  for (unsigned i = 0; i < MRI->getNumVirtRegs(); i++) {
    Register VReg = Register::index2VirtReg(i);
    if (LV.isLiveIn(VReg, MBB))
      LiveIn.insert(VReg);
  }

  for (auto &PHI : MBB.phis()) {
    for (auto U : PHI.uses()) {
      if (U.isReg()) {
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

  RegisterSet TakeVGPRS, TakeSGPRS, TakeAGPRS;
  unsigned TakeSGPRsNum = 0, TakeVGPRsNum = 0, TakeAGPRsNum = 0;

  splitByRegPressureSet(Take, TakeSGPRS, TakeSGPRsNum, TakeVGPRS, TakeVGPRsNum,
                        TakeAGPRS, TakeAGPRsNum);

  RegisterSet CandVGPRS, CandSGPRS, CandAGPRS;
  unsigned CandSGPRsNum = 0, CandVGPRsNum = 0,
           CandAGPRsNum = 0;

  splitByRegPressureSet(Cand, CandSGPRS, CandSGPRsNum,
                        CandVGPRS, CandVGPRsNum, CandAGPRS,
                        CandAGPRsNum);
  
  if (TakeSGPRsNum >= NumAvailableSGPRs) {
    NU.getSortedForInstruction(*MBB.instr_begin(), TakeSGPRS);
    Entry.ActiveSet.insert(TakeSGPRS.begin(),
                           TakeSGPRS.begin() + NumAvailableSGPRs);
  } else {
    unsigned FreeSpace = NumAvailableSGPRs - TakeSGPRsNum;
    
    Entry.ActiveSet.insert(TakeSGPRS.begin(), TakeSGPRS.end());
    NU.getSortedForInstruction(*MBB.instr_begin(), CandSGPRS);
    Entry.ActiveSet.insert(CandSGPRS.begin(),
                           CandSGPRS.begin() + FreeSpace);
  }

  formActiveSet(MBB, Take, Cand, L);
}

void AMDGPUSSASpiller::reloadAtEnd(MachineBasicBlock &MBB, Register VReg) {}

void AMDGPUSSASpiller::spillAtEnd(MachineBasicBlock &MBB, Register VReg) {}

void AMDGPUSSASpiller::reloadBefore(Register,
                                    MachineBasicBlock::iterator InsertBefore) {}

void AMDGPUSSASpiller::spillBefore(Register,
                                   MachineBasicBlock::iterator InsertBefore) {}

SmallVector<unsigned> AMDGPUSSASpiller::getLoopMaxRP(MachineLoop *L) {
  return SmallVector<unsigned>();
}

void AMDGPUSSASpiller::limit(RegisterSet &Active, RegisterSet &Spilled,
                             MachineBasicBlock::iterator I, const RegisterSet Defs) {
  MachineBasicBlock::iterator LimitPoint = I;
  RegisterSet VGPRS, SGPRS, AGPRS;
  unsigned CurSGPRsNum = 0, CurVGPRsNum = 0, CurAGPRsNum = 0;
  unsigned NumSGPRDefs = 0, NumVGPRDefs = 0, NumAGPRDefs = 0;

  splitByRegPressureSet(Active, SGPRS, CurSGPRsNum, VGPRS, CurVGPRsNum, AGPRS,
                        CurAGPRsNum);
  if (!Defs.empty()) {
    RegisterSet VGPRS, SGPRS, AGPRS;
    splitByRegPressureSet(Defs, SGPRS, NumSGPRDefs, VGPRS, NumVGPRDefs, AGPRS,
                          NumAGPRDefs);
    LimitPoint++;
  }

  if (CurSGPRsNum > NumAvailableSGPRs - NumSGPRDefs)
    limit(Active, Spilled, LimitPoint, I, SGPRS,
          CurSGPRsNum - NumAvailableSGPRs + NumSGPRDefs);

  if (CurVGPRsNum > NumAvailableVGPRs - NumVGPRDefs)
    limit(Active, Spilled, LimitPoint, I, VGPRS,
          CurVGPRsNum - NumAvailableVGPRs + NumVGPRDefs);

  if (CurAGPRsNum > NumAvailableAGPRs - NumAGPRDefs)
    limit(Active, Spilled, LimitPoint, I, AGPRS,
          CurAGPRsNum - NumAvailableAGPRs + NumAGPRDefs);
}

void AMDGPUSSASpiller::limit(RegisterSet &Active, RegisterSet &Spilled,
                             MachineBasicBlock::iterator LimitPoint,
                             MachineBasicBlock::iterator InsertionPoint,
                             RegisterSet RegClassSubset, unsigned Limit) {
  NU.getSortedForInstruction(*LimitPoint, RegClassSubset);
  RegisterSet Tmp(RegClassSubset.end() - Limit, RegClassSubset.end());
  Active.set_subtract(Tmp);
  Tmp.set_subtract(Spilled);
  for (auto R : Tmp) {
    if (!NU.isDead(*InsertionPoint, R))
      spillBefore(R, InsertionPoint);
  }
}

void AMDGPUSSASpiller::splitByRegPressureSet(
    const RegisterSet In, RegisterSet &SGPRS, unsigned &SGPRRP,
    RegisterSet &VGPRS, unsigned &VGPRRP, RegisterSet &AGPRS,
    unsigned &AGPRRP) {
  for (auto VReg : In) {
    const TargetRegisterClass *RC = TRI->getRegClass(VReg);
    unsigned Weight = TRI->getRegClassWeight(RC).RegWeight;
    const int *RPS = TRI->getRegClassPressureSets(RC);
    while (*RPS != -1) {
      if (*RPS == AMDGPU::RegisterPressureSets::SReg_32) {
        SGPRS.insert(VReg);
        SGPRRP += Weight;
        break;
      }
      if (*RPS == AMDGPU::RegisterPressureSets::VGPR_32) {
        VGPRS.insert(VReg);
        VGPRRP += Weight;
      }
      if (*RPS == AMDGPU::RegisterPressureSets::AGPR_32) {
        AGPRS.insert(VReg);
        AGPRRP += Weight;
      }
    }
  }
}

void AMDGPUSSASpiller::formActiveSet(const MachineBasicBlock &MBB,
                                     const RegisterSet Take,
                                     const RegisterSet Cand, MachineLoop *L) {
  auto &Entry = getBlockInfo(MBB);

  RegisterSet TakeVGPRS, TakeSGPRS, TakeAGPRS;
  unsigned TakeSGPRsNum = 0, TakeVGPRsNum = 0, TakeAGPRsNum = 0;

  splitByRegPressureSet(Take, TakeSGPRS, TakeSGPRsNum, TakeVGPRS, TakeVGPRsNum,
                        TakeAGPRS, TakeAGPRsNum);

  RegisterSet CandVGPRS, CandSGPRS, CandAGPRS;
  unsigned CandSGPRsNum = 0, CandVGPRsNum = 0, CandAGPRsNum = 0;

  splitByRegPressureSet(Cand, CandSGPRS, CandSGPRsNum, CandVGPRS, CandVGPRsNum,
                        CandAGPRS, CandAGPRsNum);

  if (TakeSGPRsNum >= NumAvailableSGPRs) {
    NU.getSortedForInstruction(*MBB.instr_begin(), TakeSGPRS);
    Entry.ActiveSet.insert(TakeSGPRS.begin(),
                           TakeSGPRS.begin() + NumAvailableSGPRs);
  } else {
    Entry.ActiveSet.insert(TakeSGPRS.begin(), TakeSGPRS.end());
    unsigned FreeSpace = 0;
    if (L) {
      unsigned LoopMaxSGPRRP =
          getLoopMaxRP(L)[AMDGPU::RegisterPressureSets::SReg_32];
      FreeSpace = NumAvailableSGPRs - (LoopMaxSGPRRP - CandSGPRsNum);
    } else {
      FreeSpace = NumAvailableSGPRs - TakeSGPRsNum;
    }
    NU.getSortedForInstruction(*MBB.instr_begin(), CandSGPRS);
    Entry.ActiveSet.insert(CandSGPRS.begin(), CandSGPRS.begin() + FreeSpace);
  }

  if (TakeVGPRsNum >= NumAvailableVGPRs) {
    NU.getSortedForInstruction(*MBB.instr_begin(), TakeVGPRS);
    Entry.ActiveSet.insert(TakeVGPRS.begin(),
                           TakeVGPRS.begin() + NumAvailableVGPRs);
  } else {
    Entry.ActiveSet.insert(TakeVGPRS.begin(), TakeVGPRS.end());
    unsigned FreeSpace = 0;
    if (L) {
      unsigned LoopMaxVGPRRP =
          getLoopMaxRP(L)[AMDGPU::RegisterPressureSets::VGPR_32];
      FreeSpace = NumAvailableVGPRs - (LoopMaxVGPRRP - CandVGPRsNum);
    } else {
      FreeSpace = NumAvailableVGPRs - TakeVGPRsNum;
    }
    NU.getSortedForInstruction(*MBB.instr_begin(), CandVGPRS);
    Entry.ActiveSet.insert(CandVGPRS.begin(), CandVGPRS.begin() + FreeSpace);
  }

  if (TakeAGPRsNum >= NumAvailableAGPRs) {
    NU.getSortedForInstruction(*MBB.instr_begin(), TakeAGPRS);
    Entry.ActiveSet.insert(TakeAGPRS.begin(),
                           TakeAGPRS.begin() + NumAvailableAGPRs);
  } else {
    Entry.ActiveSet.insert(TakeAGPRS.begin(), TakeAGPRS.end());
    unsigned FreeSpace = 0;
    if (L) {
      unsigned LoopMaxAGPRRP =
          getLoopMaxRP(L)[AMDGPU::RegisterPressureSets::AGPR_32];
      FreeSpace = NumAvailableAGPRs - (LoopMaxAGPRRP - CandAGPRsNum);
    } else {
      FreeSpace = NumAvailableAGPRs - TakeAGPRsNum;
    }
    NU.getSortedForInstruction(*MBB.instr_begin(), CandAGPRS);
    Entry.ActiveSet.insert(CandAGPRS.begin(), CandAGPRS.begin() + FreeSpace);
  }
}

bool AMDGPUSSASpiller::run(MachineFunction &MF) {
  init(MF);
  processFunction(MF);
  return false;
}
} // namespace

PreservedAnalyses
llvm::AMDGPUSSASpillerPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  LiveVariables &LV = MFAM.getResult<LiveVariablesAnalysis>(MF);
  MachineLoopInfo &LI = MFAM.getResult<MachineLoopAnalysis>(MF);
  MachineDominatorTree &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  AMDGPUNextUseAnalysis::Result &NU = MFAM.getResult<AMDGPUNextUseAnalysis>(MF);
  AMDGPUSSASpiller Impl(LV, LI, MDT, NU);
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
    AU.addRequired<LiveVariablesWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<AMDGPUNextUseAnalysisWrapper>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

bool AMDGPUSSASpillerLegacy::runOnMachineFunction(MachineFunction &MF) {
  LiveVariables &LV = getAnalysis<LiveVariablesWrapperPass>().getLV();
  MachineLoopInfo &LI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  AMDGPUNextUseAnalysis::Result &NU =
      getAnalysis<AMDGPUNextUseAnalysisWrapper>().getNU();
  AMDGPUSSASpiller Impl(LV, LI, MDT, NU);
  return Impl.run(MF);
}

INITIALIZE_PASS_BEGIN(AMDGPUSSASpillerLegacy, DEBUG_TYPE, "AMDGPU SSA Spiller",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
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
