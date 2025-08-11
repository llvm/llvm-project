#include "AMDGPU.h"
#include "AMDGPUSSARAUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

#include <algorithm>
#include <stack>

#include "VRegMaskPair.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rebuild-ssa"

namespace {

class AMDGPURebuildSSALegacy : public MachineFunctionPass {
  LiveIntervals *LIS;
  MachineDominatorTree *MDT;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  MachineLoopInfo *MLI;

  DenseMap<MachineOperand *, std::pair<MachineInstr *, LaneBitmask>>
      RegSeqences;

  void buildRealPHI(VNInfo *VNI, LiveInterval &LI,
                    Register OldVR);
  void splitNonPhiValue(VNInfo *VNI,
                        LiveInterval &LI, Register OldVR);
  void rewriteUses(MachineInstr *DefMI, Register OldVR,
                   LaneBitmask MaskToRewrite, Register NewVR, LiveInterval &LI,
                   VNInfo *VNI);

  typedef struct {
    Register CurName;
    LaneBitmask PrevMask;
    unsigned PrevSubRegIdx;
    MachineInstr *DefMI;
  } CurVRegInfo;

  using VRegDefStack = std::vector<CurVRegInfo>;

#ifndef NDEBUG
  void printVRegDefStack(VRegDefStack VregDefs) {
    VRegDefStack::reverse_iterator It = VregDefs.rbegin();
    dbgs() << "\n####################################\n";
    for (; It != VregDefs.rend(); ++It) {
      CurVRegInfo VRInfo = *It;
      dbgs() << printReg(VRInfo.CurName, TRI, VRInfo.PrevSubRegIdx) << "\n";
      MachineInstr *DefMI = VRInfo.DefMI;
      dbgs() << "DefMI: " << *DefMI << "\n";
      LaneBitmask DefMask = VRInfo.PrevMask;
      dbgs() << "Def mask : " << PrintLaneMask(DefMask) << "\n";
    }
    dbgs() << "####################################\n";
  }
#endif

  SetVector<VRegMaskPair> CrossBlockVRegs;
  DenseMap<VRegMaskPair, SmallPtrSet<MachineBasicBlock *, 8>> DefBlocks;
  DenseMap<VRegMaskPair, SmallPtrSet<MachineBasicBlock *, 8>> LiveInBlocks;
  DenseMap<unsigned, SetVector<VRegMaskPair>> PHINodes;
  DenseMap<MachineInstr *, VRegMaskPair> PHIMap;
  DenseSet<unsigned> DefSeen;
  DenseSet<unsigned> Renamed;
  DenseSet<unsigned> Visited;

  void collectCrossBlockVRegs(MachineFunction &MF);
  void findPHINodesPlacement(const SmallPtrSetImpl<MachineBasicBlock *> &LiveInBlocks,
                          const SmallPtrSetImpl<MachineBasicBlock *> &DefBlocks,
                          SmallVectorImpl<MachineBasicBlock *> &PHIBlocks) {
    
    IDFCalculatorBase<MachineBasicBlock, false> IDF(MDT->getBase());

    IDF.setLiveInBlocks(LiveInBlocks);
    IDF.setDefiningBlocks(DefBlocks);
    IDF.calculate(PHIBlocks);
  }

  MachineOperand &rewriteUse(MachineOperand &Op, MachineBasicBlock::iterator I,
                             MachineBasicBlock &MBB,
                             DenseMap<unsigned, VRegDefStack> VregNames) {
    // Sub-reg handling:
    // 1. if (UseMask & ~DefMask) != 0 : current Def does not define all used
    // lanes. We should search names stack for the Def that defines missed
    // lanes to construct the REG_SEQUENCE
    // 2. if (UseMask & DefMask) == 0 : current Def defines subregisters of a
    // register which are not used by the current Use. We should search names
    // stack for the corresponding sub-register def. Replace reg.subreg in Use
    // only if VReg.subreg found != current VReg.subreg in use!
    // 3. (UseMask & DefMask) == UseMask just replace the reg if the reg found
    // != current reg in Use. Take care of the subreg in Use. If (DefMask |
    // UseMask) != UseMask, i.e. current Def defines more lanes that is used
    // by the current Use, we need to calculate the corresponding subreg index
    // for the Use. DefinedLanes serves as a result of the expression
    // mentioned above. UndefSubRegs initially is set to UseMask but is
    // updated on each iteration if we are looking for the sub-regs
    // definitions to compose REG_SEQUENCE.
    bool RewriteOp = true;
    unsigned VReg = Op.getReg();
    assert(!VregNames[VReg].empty() &&
           "Error: use does not dominated by definition!\n");
    SmallVector<std::tuple<unsigned, unsigned, unsigned>> RegSeqOps;
    LaneBitmask UseMask = getOperandLaneMask(Op, TRI, MRI);
    LLVM_DEBUG(dbgs() << "Use mask : " << PrintLaneMask(UseMask)
                      << "\nLooking for appropriate definiton...\n");
    LaneBitmask UndefSubRegs = UseMask;
    LaneBitmask DefinedLanes = LaneBitmask::getNone();
    unsigned SubRegIdx = AMDGPU::NoRegister;
    Register CurVReg = AMDGPU::NoRegister;
    VRegDefStack VregDefs = VregNames[VReg];
    VRegDefStack::reverse_iterator It = VregDefs.rbegin();
    for (; It != VregDefs.rend(); ++It) {
      CurVRegInfo VRInfo = *It;
      CurVReg = VRInfo.CurName;
      MachineInstr *DefMI = VRInfo.DefMI;
      MachineOperand *DefOp = DefMI->findRegisterDefOperand(CurVReg, TRI);
      const TargetRegisterClass *RC =
          TRI->getRegClassForOperandReg(*MRI, *DefOp);
      LaneBitmask DefMask = VRInfo.PrevMask;
      LaneBitmask LanesDefinedyCurrentDef = (UndefSubRegs & DefMask) & UseMask;
      DefinedLanes |= LanesDefinedyCurrentDef;
      LLVM_DEBUG(dbgs() << "Def:\nDefMI: " << *DefMI << "\nOperand : " << *DefOp
             << "\nDef mask : " << PrintLaneMask(DefMask)
             << "\nLanes defined by current Def: "
             << PrintLaneMask(LanesDefinedyCurrentDef)
             << "\nTotal defined lanes: " << PrintLaneMask(DefinedLanes)
             << "\n");

      if (LanesDefinedyCurrentDef == UseMask) {
        // All lanes used here are defined by this def.
        if (CurVReg == VReg && Op.getSubReg() == DefOp->getSubReg()) {
          // Need nothing - bail out.
          RewriteOp = false;
          break;
        }
        SubRegIdx = DefOp->getSubReg();
        if ((DefMask & ~UseMask).any()) {
          // Definition defines more lanes then used. Need sub register
          // index;
          SubRegIdx = getSubRegIndexForLaneMask(UseMask, TRI);
        }
        break;
      }

      if (LanesDefinedyCurrentDef.any()) {
        // Current definition defines some of the lanes used here.
        unsigned DstSubReg =
            getSubRegIndexForLaneMask(LanesDefinedyCurrentDef, TRI);
        if (!DstSubReg) {
          SmallVector<unsigned> Idxs =
              getCoveringSubRegsForLaneMask(LanesDefinedyCurrentDef, RC, TRI);
          for (unsigned SubIdx : Idxs) {
            LLVM_DEBUG(dbgs() << "Matching subreg: " << SubIdx << " : "
                   << PrintLaneMask(TRI->getSubRegIndexLaneMask(SubIdx))
                   << "\n");
            RegSeqOps.push_back({CurVReg, SubIdx, SubIdx});
          }
        } else {
          unsigned SrcSubReg = (DefMask & ~LanesDefinedyCurrentDef).any()
                                   ? DstSubReg
                                   : DefOp->getSubReg();
          RegSeqOps.push_back({CurVReg, SrcSubReg, DstSubReg});
        }
        UndefSubRegs = UseMask & ~DefinedLanes;
        LLVM_DEBUG(dbgs() << "UndefSubRegs: " << PrintLaneMask(UndefSubRegs) << "\n");
        if (UndefSubRegs.none())
          break;
      } else {
        // The current definition does not define any of the lanes used
        // here. Continue to search for the definition.
        LLVM_DEBUG(dbgs() << "No lanes defined by this def!\n");
        continue;
      }
    }

    if (UndefSubRegs != UseMask && !UndefSubRegs.none()) {
      // WE haven't found all sub-regs definition. Assume undef.
      // Insert IMPLISIT_DEF

      const TargetRegisterClass *RC = TRI->getRegClassForOperandReg(*MRI, Op);
      SmallVector<unsigned> Idxs =
          getCoveringSubRegsForLaneMask(UndefSubRegs, RC, TRI);
      for (unsigned SubIdx : Idxs) {
        const TargetRegisterClass *SubRC = TRI->getSubRegisterClass(RC, SubIdx);
        Register NewVReg = MRI->createVirtualRegister(SubRC);
        BuildMI(MBB, I, I->getDebugLoc(), TII->get(AMDGPU::IMPLICIT_DEF))
            .addReg(NewVReg, RegState::Define);
        RegSeqOps.push_back({NewVReg, AMDGPU::NoRegister, SubIdx});
      }
    }

    if (!RegSeqOps.empty()) {
      // All subreg defs are found. Insert REG_SEQUENCE.
      auto *RC = TRI->getRegClassForReg(*MRI, VReg);
      CurVReg = MRI->createVirtualRegister(RC);
      auto RS = BuildMI(MBB, I, I->getDebugLoc(),
                        TII->get(AMDGPU::REG_SEQUENCE), CurVReg);
      for (auto O : RegSeqOps) {
        auto [R, SrcSubreg, DstSubreg] = O;
        RS.addReg(R, 0, SrcSubreg);
        RS.addImm(DstSubreg);
      }

      VregNames[VReg].push_back({CurVReg, MRI->getMaxLaneMaskForVReg(CurVReg),
                                 AMDGPU::NoRegister, RS});
    }

    assert(CurVReg != AMDGPU::NoRegister &&
           "Use is not dominated by definition!\n");

    if (RewriteOp) {
      LLVM_DEBUG(dbgs() << "Rewriting use: " << Op << " to "
             << printReg(CurVReg, TRI, SubRegIdx, MRI) << "\n");
      Op.setReg(CurVReg);
      Op.setSubReg(SubRegIdx);
    }
    return Op;
  }

  void renameVRegs(MachineBasicBlock &MBB,
                   DenseMap<unsigned, VRegDefStack> VregNames) {
    if (Visited.contains(MBB.getNumber()))
      return;

    for (auto &PHI : MBB.phis()) {
      MachineOperand &Op = PHI.getOperand(0);
      Register Res = Op.getReg();
      unsigned SubRegIdx = Op.getSubReg();
      const TargetRegisterClass *RC =
          SubRegIdx ? TRI->getSubRegisterClass(
                          TRI->getRegClassForReg(*MRI, Res), SubRegIdx)
                    : TRI->getRegClassForReg(*MRI, Res);
      Register NewVReg = MRI->createVirtualRegister(RC);
      Op.setReg(NewVReg);
      Op.setSubReg(AMDGPU::NoRegister);
      VregNames[Res].push_back({NewVReg,
                                SubRegIdx == AMDGPU::NoRegister
                                    ? MRI->getMaxLaneMaskForVReg(Res)
                                    : TRI->getSubRegIndexLaneMask(SubRegIdx),
                                AMDGPU::NoRegister, &PHI});
      LLVM_DEBUG(dbgs() << "\nNames stack:\n";printVRegDefStack(VregNames[Res]));
      DefSeen.insert(NewVReg);
      Renamed.insert(Res);
    }
    for (auto &I : make_range(MBB.getFirstNonPHI(), MBB.end())) {


      for (auto &Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual() &&
            Renamed.contains(Op.getReg())) {
          Op = rewriteUse(Op, I, MBB, VregNames);
        }
      }

      for (auto &Op : I.defs()) {
        if (Op.getReg().isVirtual()) {
          unsigned VReg = Op.getReg();
          if (DefSeen.contains(VReg)) {
            const TargetRegisterClass *RC =
                TRI->getRegClassForOperandReg(*MRI, Op);
            Register NewVReg = MRI->createVirtualRegister(RC);
            VregNames[VReg].push_back({NewVReg,
                                       getOperandLaneMask(Op, TRI, MRI),
                                       Op.getSubReg(), &I});
            LLVM_DEBUG(dbgs() << "\nNames stack:\n";
                       printVRegDefStack(VregNames[VReg]));

            Op.ChangeToRegister(NewVReg, true, false, false, false, false);
            Op.setSubReg(AMDGPU::NoRegister);
            LLVM_DEBUG(dbgs()
                       << "Renaming VReg: " << Register::virtReg2Index(VReg)
                       << " to " << Register::virtReg2Index(NewVReg) << "\n");
            Renamed.insert(VReg);
          } else {
            VregNames[VReg].push_back(
                {VReg, getOperandLaneMask(Op, TRI, MRI), Op.getSubReg(), &I});
            LLVM_DEBUG(dbgs() << "\nNames stack:\n";
                       printVRegDefStack(VregNames[VReg]));

            DefSeen.insert(VReg);
          }
        }
      }
    }

    Visited.insert(MBB.getNumber());

    for (auto Succ : successors(&MBB)) {
      for (auto &PHI : Succ->phis()) {
        VRegMaskPair VMP = PHIMap[&PHI];
  
        unsigned SubRegIdx = VMP.getSubReg(MRI, TRI);
        if (VregNames[VMP.getVReg()].empty()) {
          PHI.addOperand(MachineOperand::CreateReg(VMP.getVReg(), false, false,
                                                   false, false, false, false,
                                                   SubRegIdx));
        } else {
          MachineOperand Op =
              MachineOperand::CreateReg(VMP.getVReg(), false, false, false,
                                        false, false, false, SubRegIdx);
          MachineBasicBlock::iterator IP = MBB.getFirstTerminator();
          Op = rewriteUse(Op, IP, MBB, VregNames);
          PHI.addOperand(Op);
        }
        PHI.addOperand(MachineOperand::CreateMBB(&MBB));
      }
      renameVRegs(*Succ, VregNames);
    }
  }

  Printable printVMP(VRegMaskPair VMP) {
    return printReg(VMP.getVReg(), TRI, VMP.getSubReg(MRI, TRI));
  }

public:
  static char ID;
  AMDGPURebuildSSALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURebuildSSALegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitiveID(MachineDominatorsID);
    AU.addPreservedID(MachineDominatorsID);
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  };

} // end anonymous namespace

void AMDGPURebuildSSALegacy::collectCrossBlockVRegs(MachineFunction &MF) {
  for (auto &MBB : MF) {
    SetVector<VRegMaskPair> Killed;
    SetVector<VRegMaskPair> Defined;
    for (auto &I : MBB) {
      for (auto Op : I.uses()) {
        if (Op.isReg() && Op.getReg().isVirtual()) {
          VRegMaskPair VMP(Op, TRI, MRI);
          if (!Killed.contains(VMP))
            for (auto V : Defined) {
              if (V.getVReg() == VMP.getVReg()) {
                if ((V.getLaneMask() & VMP.getLaneMask()) ==
                    VMP.getLaneMask()) {
                  Killed.insert(VMP);
                  break;
                }
              }
            }
          if (!Killed.contains(VMP))
            CrossBlockVRegs.insert(VMP);
        }
      }
      for (auto Op : I.defs()) {
        if (Op.isReg() && Op.getReg().isVirtual()) {
          VRegMaskPair VMP(Op, TRI, MRI);
          Defined.insert(VMP);
          DefBlocks[VMP].insert(&MBB);
        }
      }
    }
  }
}

void AMDGPURebuildSSALegacy::buildRealPHI(VNInfo *VNI, LiveInterval &LI,
                                          Register OldVR) {
  MachineBasicBlock *DefMBB = LIS->getMBBFromIndex(VNI->def);
  SmallVector<MachineOperand> Ops;
  LaneBitmask CurrMask = LaneBitmask::getNone();
  LaneBitmask PredMask = LaneBitmask::getNone();
  LaneBitmask FullMask = MRI->getMaxLaneMaskForVReg(OldVR);
  unsigned SubRegIdx = AMDGPU::NoRegister;
  dbgs() << "\nBuild PHI for register: " << printReg(OldVR) << "\n";
  for (auto Pred : DefMBB->predecessors()) {
    dbgs() << "Pred: MBB_" << Pred->getNumber() << "\n";
    SlotIndex LastPredIdx = LIS->getMBBEndIdx(Pred);

    for (const LiveInterval::SubRange &SR : LI.subranges()) {
      // Does this sub-range contain *any* segment that refers to V ?
      if (auto V = SR.getVNInfoBefore(LastPredIdx)) {
        PredMask |= SR.LaneMask; // this lane mask is live-out of Pred
        dbgs() << "Mask : " << PrintLaneMask(SR.LaneMask) << " VNINfo: " << V
               << " id: " << V->id << "Def: " << V->def << "\n";
      }
    }

    if (!PredMask.none() && (FullMask & ~PredMask).any()) {
      // Not all lanes are merged here
      dbgs() << "Partial register merge\n";
      dbgs() << "PredMask: " << PrintLaneMask(PredMask) << "\n";
      SubRegIdx = getSubRegIndexForLaneMask(PredMask, TRI);
    } else {
      // Full register merge
      dbgs() << "Full register merge\n";
      if (PredMask.none()) {
        dbgs() << "No sub-ranges\n";
      } else {
        dbgs() << "All sub-ranges are merging. PredMask: "
               << PrintLaneMask(PredMask) << "\n";
      }
    }
    assert(CurrMask.none() || (CurrMask == PredMask));
    CurrMask = PredMask;

    Ops.push_back(
        MachineOperand::CreateReg(OldVR, 0, 0, 0, 0, 0, 0, SubRegIdx));
    Ops.push_back(MachineOperand::CreateMBB(Pred));
  }

  const TargetRegisterClass *RC =
      TRI->getRegClassForOperandReg(*MRI, Ops.front());
  
  Register DestReg =
      MRI->createVirtualRegister(RC);
  
  auto PHINode = BuildMI(*DefMBB, DefMBB->begin(), DebugLoc(),
                         TII->get(TargetOpcode::PHI), DestReg)
                     .add(ArrayRef(Ops));

  MachineInstr *PHI = PHINode.getInstr();
  LIS->InsertMachineInstrInMaps(*PHI);

  rewriteUses(PHI, OldVR, CurrMask.none() ? FullMask : CurrMask, DestReg, LI,
              VNI);
  LIS->createAndComputeVirtRegInterval(DestReg);
}

void AMDGPURebuildSSALegacy::splitNonPhiValue(VNInfo *VNI, LiveInterval &LI,
                                              Register OldVR) {
  MachineInstr *DefMI = LIS->getInstructionFromIndex(VNI->def);
  int DefIdx = DefMI->findRegisterDefOperandIdx(OldVR, TRI, false, true);
  MachineOperand &MO = DefMI->getOperand(DefIdx);
  unsigned SubRegIdx = MO.getSubReg();
  LaneBitmask Mask = SubRegIdx ? TRI->getSubRegIndexLaneMask(SubRegIdx)
                               : MRI->getMaxLaneMaskForVReg(MO.getReg());
  const TargetRegisterClass *RC = TRI->getRegClassForOperandReg(*MRI, MO);
  Register NewVR = MRI->createVirtualRegister(RC);
  MO.setReg(NewVR);
  MO.setSubReg(AMDGPU::NoRegister);
  MO.setIsUndef(false);
  LIS->ReplaceMachineInstrInMaps(*DefMI, *DefMI);
  rewriteUses(DefMI, OldVR, Mask, NewVR, LI, VNI);

  LIS->createAndComputeVirtRegInterval(NewVR);
}

void AMDGPURebuildSSALegacy::rewriteUses(MachineInstr *DefMI, Register OldVR,
                                         LaneBitmask MaskToRewrite, Register NewVR,
                                         LiveInterval &LI, VNInfo *VNI) {
  for (MachineOperand &MO :
       llvm::make_early_inc_range(MRI->use_operands(OldVR))) {
    MachineInstr *UseMI = MO.getParent();
    if (DefMI == UseMI)
      continue;
    SlotIndex UseIdx = LIS->getInstructionIndex(*UseMI);

    if (UseMI->getParent() == DefMI->getParent()) {
      SlotIndex DefIdx = LIS->getInstructionIndex(*DefMI);

      if (DefIdx >= UseIdx) {
        if (MLI->isLoopHeader(UseMI->getParent()) && UseMI->isPHI()) {
          unsigned OpIdx = UseMI->getOperandNo(&MO);
          MachineBasicBlock *Pred = UseMI->getOperand(++OpIdx).getMBB();
          SlotIndex PredEnd = LIS->getMBBEndIdx(Pred);
          VNInfo *InV = LI.getVNInfoBefore(PredEnd);

          if (InV != VNI)
            continue;
        } else
          continue;
      }
    } else {
      if (UseMI->isPHI()) {
        unsigned OpIdx = UseMI->getOperandNo(&MO);
        MachineBasicBlock *Pred = UseMI->getOperand(++OpIdx).getMBB();
        SlotIndex PredEnd = LIS->getMBBEndIdx(Pred);
        VNInfo *InV = LI.getVNInfoBefore(PredEnd);

        if (InV != VNI)
          continue;
      } else if (!MDT->dominates(DefMI->getParent(), UseMI->getParent()))
        continue;
    }
    const TargetRegisterClass *NewRC = TRI->getRegClassForReg(*MRI, NewVR);
    const TargetRegisterClass *OpRC = TRI->getRegClassForOperandReg(*MRI, MO);
    LaneBitmask OpMask = MRI->getMaxLaneMaskForVReg(MO.getReg());
    if (MO.getSubReg()) {
      OpMask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
    }
    if ((OpMask & MaskToRewrite).none())
      continue;
    if (isOfRegClass(getRegSubRegPair(MO), *NewRC, *MRI) &&
        OpMask == MaskToRewrite) {
      MO.setReg(NewVR);
      MO.setSubReg(AMDGPU::NoRegister);
    } else {
      if ((OpMask & ~MaskToRewrite).any()) {
        // super-register use
        LaneBitmask Mask = LaneBitmask::getNone();
        // We need to explicitly inform LIS that the subreg is live up to the
        // REG_SEQUENCE
        LaneBitmask SubRangeToExtend = LaneBitmask::getNone();
        Register DestReg = MRI->createVirtualRegister(OpRC);
        MachineBasicBlock::iterator IP(UseMI);
        if (UseMI->isPHI()) {
          unsigned OpIdx = UseMI->getOperandNo(&MO);
          MachineBasicBlock *Pred = UseMI->getOperand(++OpIdx).getMBB();
          IP = Pred->getFirstTerminator();
        }
        auto RS = BuildMI(*IP->getParent(), IP, IP->getDebugLoc(),
                          TII->get(TargetOpcode::REG_SEQUENCE), DestReg);
        for (const LiveInterval::SubRange &SR : LI.subranges()) {
          // Does this sub-range contain *any* segment that refers to V ?
          if (SR.getVNInfoAt(UseIdx)) {
            Mask = SR.LaneMask; // this lane mask is live-out of Pred
            dbgs() << PrintLaneMask(Mask) << "\n";
            unsigned SubRegIdx = getSubRegIndexForLaneMask(Mask, TRI);
            if (Mask == MaskToRewrite)
              RS.addReg(NewVR).addImm(SubRegIdx);
            else {
              RS.addReg(OldVR, 0, SubRegIdx).addImm(SubRegIdx);
              // We only save the mask for those sub-regs which have not been
              // rewriten. For the rewiritten we will call the
              // createAndComputeLiveREgInterval afterwords.
              SubRangeToExtend = SR.LaneMask;
            }
          }
        }
        auto RSIdx = LIS->InsertMachineInstrInMaps(*RS);
        LIS->extendToIndices(LI, ArrayRef(RSIdx));
        for (auto &SR : LI.subranges()) {
          if (SR.LaneMask == SubRangeToExtend)
            LIS->extendToIndices(SR, ArrayRef(RSIdx));
          }
          MO.setReg(RS->getOperand(0).getReg());
        } else if ((OpMask & MaskToRewrite) == OpMask) {
          // sub-register use
          if (UseMI->isPHI()) {
            unsigned OpIdx = UseMI->getOperandNo(&MO);
            MachineBasicBlock *Pred = UseMI->getOperand(++OpIdx).getMBB();
            SlotIndex PredEnd = LIS->getMBBEndIdx(Pred);
            VNInfo *InV = LI.getVNInfoBefore(PredEnd);

            if (InV != VNI)
              continue;
          }
          unsigned SubRegIdx = MO.getSubReg();
          assert(SubRegIdx != AMDGPU::NoRegister &&
                 "Sub-register must not be zero");
          MO.setReg(NewVR);
          MO.setSubReg(SubRegIdx);
        }
    }
  }
}

bool AMDGPURebuildSSALegacy::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  if (MRI->isSSA())
    return false;

  CrossBlockVRegs.clear();
  DefBlocks.clear();
  LiveInBlocks.clear();
  PHINodes.clear();
  DefSeen.clear();
  Renamed.clear();
  Visited.clear();

  DenseSet<Register> Processed;

  for (auto &B : MF) {
    for (auto &I : B) {
      for (auto Def : I.defs()) {
        if (Def.isReg() && Def.getReg().isVirtual()) {
          Register VReg = Def.getReg();
          if (!LIS->hasInterval(VReg) || !Processed.insert(VReg).second)
            continue;
          auto &LI = LIS->getInterval(VReg);
          if (LI.getNumValNums() == 1)
            continue;
          
          SmallVector<VNInfo *, 8> WorkList;
          for (VNInfo *V : LI.vnis())
          // for (const LiveInterval::SubRange &SR : LI.subranges())
            // for (auto V : SR.vnis())
              if (V && !V->isUnused())
                WorkList.push_back(V);

          auto DomKey = [&](VNInfo *V) {
            MachineBasicBlock *BB = LIS->getMBBFromIndex(V->def);
            // DomTree preorder index (DFS number) – cheaper than repeated
            // dominates()
            static DenseMap<MachineBasicBlock *, unsigned> Num;
            if (Num.empty()) {
              unsigned N = 0;
              for (auto *Node : depth_first(MDT->getRootNode()))
                Num[Node->getBlock()] = N++;
            }
            return std::pair{Num[BB], V->def}; // tie-break with SlotIndex
          };

          llvm::sort(WorkList, [&](VNInfo *A, VNInfo *B) {
            return DomKey(A) < DomKey(B); // strict weak order
          });

          for (auto V : WorkList) {
            dbgs() << "id: " << V->id << " Def: " << V->def
                   << " isPHI: " << V->isPHIDef() << "\n";
          }
         

          // --- the root is now Work[0] ---
          VNInfo *Root = WorkList.front(); // dominator of all others
          // 2. stable-partition: PHIs (except root) to the front
          auto IsPhi = [&](VNInfo *V) { return V != Root && V->isPHIDef(); };
          auto Mid =
              std::stable_partition(WorkList.begin(), WorkList.end(), IsPhi);

          // 3. Phase A: build real PHIs, leave incoming defs unchanged
          auto PHISlice =
              llvm::ArrayRef(WorkList).take_front(Mid - WorkList.begin());
          for (auto It = PHISlice.rbegin(); It != PHISlice.rend(); ++It) {
            // Add PHIs in post-dominating order
            buildRealPHI(*It, LI, VReg);
          }

          // 4. Phase B: split the remaining VNIs
          for (VNInfo *VNI : llvm::ArrayRef(WorkList).slice(Mid - WorkList.begin())) {
            if (VNI == Root)
              continue;            // never touch the dominating root
            splitNonPhiValue(VNI, LI, VReg);
          }

          // 5. single clean-up
          // LIS->shrinkToUses(&LI);
          LI.RenumberValues();
        }
      }
    }
  }

  Processed.clear();

  // // Collect all cross-block virtual registers.
  // // This includes registers that are live-in to the function, and registers
  // // that are defined in multiple blocks.
  // // We will insert PHI nodes for these registers.
  // collectCrossBlockVRegs(MF);

  // LLVM_DEBUG(dbgs() << "##### Virt regs live cross block ##################\n";
  //            for (auto VMP : CrossBlockVRegs) { dbgs() << printVMP(VMP) << " "; });

  // for (auto VMP : CrossBlockVRegs) {
  //   SmallVector<MachineBasicBlock *> PHIBlocks;
  //   LiveInterval &LI = LIS->getInterval(VMP.getVReg());
  //   if (LI.hasSubRanges()) {
  //     for (const LiveInterval::SubRange &SR : LI.subranges()) {
  //       LaneBitmask Mask = SR.LaneMask;
  //       if ((Mask & VMP.getLaneMask()) == VMP.getLaneMask()) {
  //         for (auto &MBB : MF) {
  //           if (SR.liveAt(LIS->getMBBStartIdx(&MBB)))
  //             LiveInBlocks[VMP].insert(&MBB);
  //         }
  //       }
  //     }
  //   } else {
  //     for (auto &MBB : MF) {
  //       if (LI.liveAt(LIS->getMBBStartIdx(&MBB)))
  //         LiveInBlocks[VMP].insert(&MBB);
  //     }
  //   }

  //   SmallPtrSet<MachineBasicBlock *, 8> Defs;
  //   for(auto E : DefBlocks) {
  //     auto V = E.first;
  //     if (V.getVReg() == VMP.getVReg()) {
  //       if ((V.getLaneMask() & VMP.getLaneMask()) == VMP.getLaneMask()) {
  //         Defs.insert(E.second.begin(), E.second.end());
  //       }
  //     }
  //   }

  //   LLVM_DEBUG(
  //       dbgs() << "findPHINodesPlacement input:\nVreg: "
  //              << printVMP(VMP)
  //              << "\n";
  //       dbgs() << "Def Blocks: \n"; for (auto MBB
  //                                        : Defs) {
  //         dbgs() << "MBB_" << MBB->getNumber() << " ";
  //       } dbgs() << "\nLiveIn Blocks: \n";
  //       for (auto MBB
  //            : LiveInBlocks[VMP]) {
  //         dbgs() << "MBB_" << MBB->getNumber() << " ";
  //       } dbgs()
  //       << "\n");

  //   findPHINodesPlacement(LiveInBlocks[VMP], Defs, PHIBlocks);
  //   LLVM_DEBUG(dbgs() << "\nBlocks to insert PHI nodes:\n"; for (auto MBB
  //                                                                : PHIBlocks) {
  //     dbgs() << "MBB_" << MBB->getNumber() << " ";
  //   } dbgs() << "\n");
  //   for (auto MBB : PHIBlocks) {
  //     if (!PHINodes[MBB->getNumber()].contains(VMP)) {
  //       // Insert PHI for VReg. Don't use new VReg here as we'll replace them
  //       // in renaming phase.
  //       printVMP(VMP);
  //       auto PHINode =
  //           BuildMI(*MBB, MBB->begin(), DebugLoc(), TII->get(TargetOpcode::PHI))
  //               .addReg(VMP.getVReg(), RegState::Define, VMP.getSubReg(MRI, TRI));
  //       PHINodes[MBB->getNumber()].insert(VMP);
  //       PHIMap[PHINode] = VMP;
  //     }
  //   }
  // }

  //   // Rename virtual registers in the basic block.
  // DenseMap<unsigned, VRegDefStack> VregNames;
  // renameVRegs(MF.front(), VregNames);
  MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
  MF.getProperties().reset(MachineFunctionProperties::Property ::NoPHIs);
  MF.verify();
  return MRI->isSSA();
}

char AMDGPURebuildSSALegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                    false, false)

// Legacy PM registration
FunctionPass *llvm::createAMDGPURebuildSSALegacyPass() {
  return new AMDGPURebuildSSALegacy();
}

PreservedAnalyses
llvm::AMDGPURebuildSSAPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  AMDGPURebuildSSALegacy Impl;
  bool Changed = Impl.runOnMachineFunction(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  // TODO: We could detect CFG changed.
  auto PA = getMachineFunctionPassPreservedAnalyses();
  return PA;
}

llvm::PassPluginLibraryInfo getAMDGPURebuildSSAPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AMDGPURebuildSSA", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, MachineFunctionPassManager &PM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "amdgpu-rebuild-ssa") {
                    PM.addPass(AMDGPURebuildSSAPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// Expose the pass to LLVM’s pass manager infrastructure
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAMDGPURebuildSSAPassPluginInfo();
}