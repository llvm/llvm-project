//===-- SILowerSGPRSPills.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handle SGPR spills. This pass takes the place of PrologEpilogInserter for all
// SGPR spills, so must insert CSR SGPR spills as well as expand them.
//
// This pass must never create new SGPR virtual registers.
//
// FIXME: Must stop RegScavenger spills in later passes.
//
//===----------------------------------------------------------------------===//

#include "SILowerSGPRSpills.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"

using namespace llvm;

#define DEBUG_TYPE "si-lower-sgpr-spills"

using MBBVector = SmallVector<MachineBasicBlock *, 4>;

namespace {

static cl::opt<unsigned> MaxNumVGPRsForWwmAllocation(
    "amdgpu-num-vgprs-for-wwm-alloc",
    cl::desc("Max num VGPRs for whole-wave register allocation."),
    cl::ReallyHidden, cl::init(10));

class SILowerSGPRSpills {
private:
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  LiveIntervals *LIS = nullptr;
  SlotIndexes *Indexes = nullptr;
  MachineDominatorTree *MDT = nullptr;

  // Save and Restore blocks of the current function. Typically there is a
  // single save block, unless Windows EH funclets are involved.
  MBBVector SaveBlocks;
  MBBVector RestoreBlocks;

public:
  SILowerSGPRSpills(LiveIntervals *LIS, SlotIndexes *Indexes,
                    MachineDominatorTree *MDT)
      : LIS(LIS), Indexes(Indexes), MDT(MDT) {}
  bool run(MachineFunction &MF);
  void calculateSaveRestoreBlocks(MachineFunction &MF);
  bool spillCalleeSavedRegs(MachineFunction &MF,
                            SmallVectorImpl<int> &CalleeSavedFIs);
  void updateLaneVGPRDomInstr(
      int FI, MachineBasicBlock *MBB, MachineBasicBlock::iterator InsertPt,
      DenseMap<Register, MachineBasicBlock::iterator> &LaneVGPRDomInstr);
  void determineRegsForWWMAllocation(MachineFunction &MF, BitVector &RegMask);
};

class SILowerSGPRSpillsLegacy : public MachineFunctionPass {
public:
  static char ID;

  SILowerSGPRSpillsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getClearedProperties() const override {
    // SILowerSGPRSpills introduces new Virtual VGPRs for spilling SGPRs.
    return MachineFunctionProperties().setIsSSA().setNoVRegs();
  }
};

} // end anonymous namespace

char SILowerSGPRSpillsLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(SILowerSGPRSpillsLegacy, DEBUG_TYPE,
                      "SI lower SGPR spill instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_END(SILowerSGPRSpillsLegacy, DEBUG_TYPE,
                    "SI lower SGPR spill instructions", false, false)

char &llvm::SILowerSGPRSpillsLegacyID = SILowerSGPRSpillsLegacy::ID;

static bool isLiveIntoMBB(MCRegister Reg, MachineBasicBlock &MBB,
                          const TargetRegisterInfo *TRI) {
  for (MCRegAliasIterator R(Reg, TRI, true); R.isValid(); ++R) {
    if (MBB.isLiveIn(*R)) {
      return true;
    }
  }
  return false;
}

/// Insert spill code for the callee-saved registers used in the function.
static void insertCSRSaves(MachineBasicBlock &SaveBlock,
                           ArrayRef<CalleeSavedInfo> CSI, SlotIndexes *Indexes,
                           LiveIntervals *LIS) {
  MachineFunction &MF = *SaveBlock.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *RI = ST.getRegisterInfo();

  MachineBasicBlock::iterator I = SaveBlock.begin();
  if (!TFI->spillCalleeSavedRegisters(SaveBlock, I, CSI, TRI)) {
    for (const CalleeSavedInfo &CS : CSI) {
      // Insert the spill to the stack frame.
      MCRegister Reg = CS.getReg();

      MachineInstrSpan MIS(I, &SaveBlock);
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(
          Reg, Reg == RI->getReturnAddressReg(MF) ? MVT::i64 : MVT::i32);

      // If this value was already livein, we probably have a direct use of the
      // incoming register value, so don't kill at the spill point. This happens
      // since we pass some special inputs (workgroup IDs) in the callee saved
      // range.
      const bool IsLiveIn = isLiveIntoMBB(Reg, SaveBlock, TRI);
      TII.storeRegToStackSlot(SaveBlock, I, Reg, !IsLiveIn, CS.getFrameIdx(),
                              RC, TRI, Register());

      if (Indexes) {
        assert(std::distance(MIS.begin(), I) == 1);
        MachineInstr &Inst = *std::prev(I);
        Indexes->insertMachineInstrInMaps(Inst);
      }

      if (LIS)
        LIS->removeAllRegUnitsForPhysReg(Reg);
    }
  } else {
    // TFI doesn't update Indexes and LIS, so we have to do it separately.
    if (Indexes)
      Indexes->repairIndexesInRange(&SaveBlock, SaveBlock.begin(), I);

    if (LIS)
      for (const CalleeSavedInfo &CS : CSI)
        LIS->removeAllRegUnitsForPhysReg(CS.getReg());
  }
}

/// Insert restore code for the callee-saved registers used in the function.
static void insertCSRRestores(MachineBasicBlock &RestoreBlock,
                              MutableArrayRef<CalleeSavedInfo> CSI,
                              SlotIndexes *Indexes, LiveIntervals *LIS) {
  MachineFunction &MF = *RestoreBlock.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  // Restore all registers immediately before the return and any
  // terminators that precede it.
  MachineBasicBlock::iterator I = RestoreBlock.getFirstTerminator();
  const MachineBasicBlock::iterator BeforeRestoresI =
      I == RestoreBlock.begin() ? I : std::prev(I);

  // FIXME: Just emit the readlane/writelane directly
  if (!TFI->restoreCalleeSavedRegisters(RestoreBlock, I, CSI, TRI)) {
    for (const CalleeSavedInfo &CI : reverse(CSI)) {
      // Insert in reverse order.  loadRegFromStackSlot can insert
      // multiple instructions.
      TFI->restoreCalleeSavedRegister(RestoreBlock, I, CI, &TII, TRI);

      if (Indexes) {
        MachineInstr &Inst = *std::prev(I);
        Indexes->insertMachineInstrInMaps(Inst);
      }

      if (LIS)
        LIS->removeAllRegUnitsForPhysReg(CI.getReg());
    }
  } else {
    // TFI doesn't update Indexes and LIS, so we have to do it separately.
    if (Indexes)
      Indexes->repairIndexesInRange(&RestoreBlock, BeforeRestoresI,
                                    RestoreBlock.getFirstTerminator());

    if (LIS)
      for (const CalleeSavedInfo &CS : CSI)
        LIS->removeAllRegUnitsForPhysReg(CS.getReg());
  }
}

/// Compute the sets of entry and return blocks for saving and restoring
/// callee-saved registers, and placing prolog and epilog code.
void SILowerSGPRSpills::calculateSaveRestoreBlocks(MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // Even when we do not change any CSR, we still want to insert the
  // prologue and epilogue of the function.
  // So set the save points for those.

  // Use the points found by shrink-wrapping, if any.
  if (!MFI.getSavePoints().empty()) {
    assert(MFI.getSavePoints().size() == 1 &&
           "Multiple save points not yet supported!");
    SaveBlocks.push_back(MFI.getSavePoints().front());
    assert(MFI.getRestorePoints().size() == 1 &&
           "Multiple restore points not yet supported!");
    MachineBasicBlock *RestoreBlock = MFI.getRestorePoints().front();
    // If RestoreBlock does not have any successor and is not a return block
    // then the end point is unreachable and we do not need to insert any
    // epilogue.
    if (!RestoreBlock->succ_empty() || RestoreBlock->isReturnBlock())
      RestoreBlocks.push_back(RestoreBlock);
    return;
  }

  // Save refs to entry and return blocks.
  SaveBlocks.push_back(&MF.front());
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.isEHFuncletEntry())
      SaveBlocks.push_back(&MBB);
    if (MBB.isReturnBlock())
      RestoreBlocks.push_back(&MBB);
  }
}

// TODO: To support shrink wrapping, this would need to copy
// PrologEpilogInserter's updateLiveness.
static void updateLiveness(MachineFunction &MF, ArrayRef<CalleeSavedInfo> CSI) {
  MachineBasicBlock &EntryBB = MF.front();

  for (const CalleeSavedInfo &CSIReg : CSI)
    EntryBB.addLiveIn(CSIReg.getReg());
  EntryBB.sortUniqueLiveIns();
}

bool SILowerSGPRSpills::spillCalleeSavedRegs(
    MachineFunction &MF, SmallVectorImpl<int> &CalleeSavedFIs) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = MF.getFunction();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIFrameLowering *TFI = ST.getFrameLowering();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  RegScavenger *RS = nullptr;

  // Determine which of the registers in the callee save list should be saved.
  BitVector SavedRegs;
  TFI->determineCalleeSavesSGPR(MF, SavedRegs, RS);

  // Add the code to save and restore the callee saved registers.
  if (!F.hasFnAttribute(Attribute::Naked)) {
    // FIXME: This is a lie. The CalleeSavedInfo is incomplete, but this is
    // necessary for verifier liveness checks.
    MFI.setCalleeSavedInfoValid(true);

    std::vector<CalleeSavedInfo> CSI;
    const MCPhysReg *CSRegs = MRI.getCalleeSavedRegs();

    for (unsigned I = 0; CSRegs[I]; ++I) {
      MCRegister Reg = CSRegs[I];

      if (SavedRegs.test(Reg)) {
        const TargetRegisterClass *RC =
          TRI->getMinimalPhysRegClass(Reg, MVT::i32);
        int JunkFI = MFI.CreateStackObject(TRI->getSpillSize(*RC),
                                           TRI->getSpillAlign(*RC), true);

        CSI.emplace_back(Reg, JunkFI);
        CalleeSavedFIs.push_back(JunkFI);
      }
    }

    if (!CSI.empty()) {
      for (MachineBasicBlock *SaveBlock : SaveBlocks)
        insertCSRSaves(*SaveBlock, CSI, Indexes, LIS);

      // Add live ins to save blocks.
      assert(SaveBlocks.size() == 1 && "shrink wrapping not fully implemented");
      updateLiveness(MF, CSI);

      for (MachineBasicBlock *RestoreBlock : RestoreBlocks)
        insertCSRRestores(*RestoreBlock, CSI, Indexes, LIS);
      return true;
    }
  }

  return false;
}

void SILowerSGPRSpills::updateLaneVGPRDomInstr(
    int FI, MachineBasicBlock *MBB, MachineBasicBlock::iterator InsertPt,
    DenseMap<Register, MachineBasicBlock::iterator> &LaneVGPRDomInstr) {
  // For the Def of a virtual LaneVPGR to dominate all its uses, we should
  // insert an IMPLICIT_DEF before the dominating spill. Switching to a
  // depth first order doesn't really help since the machine function can be in
  // the unstructured control flow post-SSA. For each virtual register, hence
  // finding the common dominator to get either the dominating spill or a block
  // dominating all spills.
  SIMachineFunctionInfo *FuncInfo =
      MBB->getParent()->getInfo<SIMachineFunctionInfo>();
  ArrayRef<SIRegisterInfo::SpilledReg> VGPRSpills =
      FuncInfo->getSGPRSpillToVirtualVGPRLanes(FI);
  Register PrevLaneVGPR;
  for (auto &Spill : VGPRSpills) {
    if (PrevLaneVGPR == Spill.VGPR)
      continue;

    PrevLaneVGPR = Spill.VGPR;
    auto I = LaneVGPRDomInstr.find(Spill.VGPR);
    if (Spill.Lane == 0 && I == LaneVGPRDomInstr.end()) {
      // Initially add the spill instruction itself for Insertion point.
      LaneVGPRDomInstr[Spill.VGPR] = InsertPt;
    } else {
      assert(I != LaneVGPRDomInstr.end());
      auto PrevInsertPt = I->second;
      MachineBasicBlock *DomMBB = PrevInsertPt->getParent();
      if (DomMBB == MBB) {
        // The insertion point earlier selected in a predecessor block whose
        // spills are currently being lowered. The earlier InsertPt would be
        // the one just before the block terminator and it should be changed
        // if we insert any new spill in it.
        if (MDT->dominates(&*InsertPt, &*PrevInsertPt))
          I->second = InsertPt;

        continue;
      }

      // Find the common dominator block between PrevInsertPt and the
      // current spill.
      DomMBB = MDT->findNearestCommonDominator(DomMBB, MBB);
      if (DomMBB == MBB)
        I->second = InsertPt;
      else if (DomMBB != PrevInsertPt->getParent())
        I->second = &(*DomMBB->getFirstTerminator());
    }
  }
}

void SILowerSGPRSpills::determineRegsForWWMAllocation(MachineFunction &MF,
                                                      BitVector &RegMask) {
  // Determine an optimal number of VGPRs for WWM allocation. The complement
  // list will be available for allocating other VGPR virtual registers.
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  BitVector ReservedRegs = TRI->getReservedRegs(MF);
  BitVector NonWwmAllocMask(TRI->getNumRegs());
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  // FIXME: MaxNumVGPRsForWwmAllocation might need to be adjusted in the future
  // to have a balanced allocation between WWM values and per-thread vector
  // register operands.
  unsigned NumRegs = MaxNumVGPRsForWwmAllocation;
  NumRegs =
      std::min(static_cast<unsigned>(MFI->getSGPRSpillVGPRs().size()), NumRegs);

  auto [MaxNumVGPRs, MaxNumAGPRs] = ST.getMaxNumVectorRegs(MF.getFunction());
  // Try to use the highest available registers for now. Later after
  // vgpr-regalloc, they can be shifted to the lowest range.
  unsigned I = 0;
  for (unsigned Reg = AMDGPU::VGPR0 + MaxNumVGPRs - 1;
       (I < NumRegs) && (Reg >= AMDGPU::VGPR0); --Reg) {
    if (!ReservedRegs.test(Reg) &&
        !MRI.isPhysRegUsed(Reg, /*SkipRegMaskTest=*/true)) {
      TRI->markSuperRegs(RegMask, Reg);
      ++I;
    }
  }

  if (I != NumRegs) {
    // Reserve an arbitrary register and report the error.
    TRI->markSuperRegs(RegMask, AMDGPU::VGPR0);
    MF.getFunction().getContext().emitError(
        "cannot find enough VGPRs for wwm-regalloc");
  }
}

bool SILowerSGPRSpillsLegacy::runOnMachineFunction(MachineFunction &MF) {
  auto *LISWrapper = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
  LiveIntervals *LIS = LISWrapper ? &LISWrapper->getLIS() : nullptr;
  auto *SIWrapper = getAnalysisIfAvailable<SlotIndexesWrapperPass>();
  SlotIndexes *Indexes = SIWrapper ? &SIWrapper->getSI() : nullptr;
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  return SILowerSGPRSpills(LIS, Indexes, MDT).run(MF);
}

bool SILowerSGPRSpills::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();

  assert(SaveBlocks.empty() && RestoreBlocks.empty());

  // First, expose any CSR SGPR spills. This is mostly the same as what PEI
  // does, but somewhat simpler.
  calculateSaveRestoreBlocks(MF);
  SmallVector<int> CalleeSavedFIs;
  bool HasCSRs = spillCalleeSavedRegs(MF, CalleeSavedFIs);

  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();

  if (!MFI.hasStackObjects() && !HasCSRs) {
    SaveBlocks.clear();
    RestoreBlocks.clear();
    return false;
  }

  bool MadeChange = false;
  bool SpilledToVirtVGPRLanes = false;

  // TODO: CSR VGPRs will never be spilled to AGPRs. These can probably be
  // handled as SpilledToReg in regular PrologEpilogInserter.
  const bool HasSGPRSpillToVGPR = TRI->spillSGPRToVGPR() &&
                                  (HasCSRs || FuncInfo->hasSpilledSGPRs());
  if (HasSGPRSpillToVGPR) {
    // Process all SGPR spills before frame offsets are finalized. Ideally SGPRs
    // are spilled to VGPRs, in which case we can eliminate the stack usage.
    //
    // This operates under the assumption that only other SGPR spills are users
    // of the frame index.

    // To track the spill frame indices handled in this pass.
    BitVector SpillFIs(MFI.getObjectIndexEnd(), false);

    // To track the IMPLICIT_DEF insertion point for the lane vgprs.
    DenseMap<Register, MachineBasicBlock::iterator> LaneVGPRDomInstr;

    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
        if (!TII->isSGPRSpill(MI))
          continue;

        if (MI.getOperand(0).isUndef()) {
          if (Indexes)
            Indexes->removeMachineInstrFromMaps(MI);
          MI.eraseFromParent();
          continue;
        }

        int FI = TII->getNamedOperand(MI, AMDGPU::OpName::addr)->getIndex();
        assert(MFI.getStackID(FI) == TargetStackID::SGPRSpill);

        bool IsCalleeSaveSGPRSpill = llvm::is_contained(CalleeSavedFIs, FI);
        if (IsCalleeSaveSGPRSpill) {
          // Spill callee-saved SGPRs into physical VGPR lanes.

          // TODO: This is to ensure the CFIs are static for efficient frame
          // unwinding in the debugger. Spilling them into virtual VGPR lanes
          // involve regalloc to allocate the physical VGPRs and that might
          // cause intermediate spill/split of such liveranges for successful
          // allocation. This would result in broken CFI encoding unless the
          // regalloc aware CFI generation to insert new CFIs along with the
          // intermediate spills is implemented. There is no such support
          // currently exist in the LLVM compiler.
          if (FuncInfo->allocateSGPRSpillToVGPRLane(
                  MF, FI, /*SpillToPhysVGPRLane=*/true)) {
            bool Spilled = TRI->eliminateSGPRToVGPRSpillFrameIndex(
                MI, FI, nullptr, Indexes, LIS, true);
            if (!Spilled)
              llvm_unreachable(
                  "failed to spill SGPR to physical VGPR lane when allocated");
          }
        } else {
          MachineInstrSpan MIS(&MI, &MBB);
          if (FuncInfo->allocateSGPRSpillToVGPRLane(MF, FI)) {
            bool Spilled = TRI->eliminateSGPRToVGPRSpillFrameIndex(
                MI, FI, nullptr, Indexes, LIS);
            if (!Spilled)
              llvm_unreachable(
                  "failed to spill SGPR to virtual VGPR lane when allocated");
            SpillFIs.set(FI);
            updateLaneVGPRDomInstr(FI, &MBB, MIS.begin(), LaneVGPRDomInstr);
            SpilledToVirtVGPRLanes = true;
          }
        }
      }
    }

    for (auto Reg : FuncInfo->getSGPRSpillVGPRs()) {
      auto InsertPt = LaneVGPRDomInstr[Reg];
      // Insert the IMPLICIT_DEF at the identified points.
      MachineBasicBlock &Block = *InsertPt->getParent();
      DebugLoc DL = Block.findDebugLoc(InsertPt);
      auto MIB =
          BuildMI(Block, *InsertPt, DL, TII->get(AMDGPU::IMPLICIT_DEF), Reg);

      // Add WWM flag to the virtual register.
      FuncInfo->setFlag(Reg, AMDGPU::VirtRegFlag::WWM_REG);

      // Set SGPR_SPILL asm printer flag
      MIB->setAsmPrinterFlag(AMDGPU::SGPR_SPILL);
      if (LIS) {
        LIS->InsertMachineInstrInMaps(*MIB);
        LIS->createAndComputeVirtRegInterval(Reg);
      }
    }

    // Determine the registers for WWM allocation and also compute the register
    // mask for non-wwm VGPR allocation.
    if (FuncInfo->getSGPRSpillVGPRs().size()) {
      BitVector WwmRegMask(TRI->getNumRegs());

      determineRegsForWWMAllocation(MF, WwmRegMask);

      BitVector NonWwmRegMask(WwmRegMask);
      NonWwmRegMask.flip().clearBitsNotInMask(TRI->getAllVGPRRegMask());

      // The complement set will be the registers for non-wwm (per-thread) vgpr
      // allocation.
      FuncInfo->updateNonWWMRegMask(NonWwmRegMask);
    }

    for (MachineBasicBlock &MBB : MF) {
      // FIXME: The dead frame indices are replaced with a null register from
      // the debug value instructions. We should instead, update it with the
      // correct register value. But not sure the register value alone is
      // adequate to lower the DIExpression. It should be worked out later.
      for (MachineInstr &MI : MBB) {
        if (MI.isDebugValue()) {
          uint32_t StackOperandIdx = MI.isDebugValueList() ? 2 : 0;
          if (MI.getOperand(StackOperandIdx).isFI() &&
              !MFI.isFixedObjectIndex(
                  MI.getOperand(StackOperandIdx).getIndex()) &&
              SpillFIs[MI.getOperand(StackOperandIdx).getIndex()]) {
            MI.getOperand(StackOperandIdx)
                .ChangeToRegister(Register(), false /*isDef*/);
          }
        }
      }
    }

    // All those frame indices which are dead by now should be removed from the
    // function frame. Otherwise, there is a side effect such as re-mapping of
    // free frame index ids by the later pass(es) like "stack slot coloring"
    // which in turn could mess-up with the book keeping of "frame index to VGPR
    // lane".
    FuncInfo->removeDeadFrameIndices(MFI, /*ResetSGPRSpillStackIDs*/ false);

    MadeChange = true;
  }

  if (SpilledToVirtVGPRLanes) {
    const TargetRegisterClass *RC = TRI->getWaveMaskRegClass();
    // Shift back the reserved SGPR for EXEC copy into the lowest range.
    // This SGPR is reserved to handle the whole-wave spill/copy operations
    // that might get inserted during vgpr regalloc.
    Register UnusedLowSGPR = TRI->findUnusedRegister(MRI, RC, MF);
    if (UnusedLowSGPR && TRI->getHWRegIndex(UnusedLowSGPR) <
                             TRI->getHWRegIndex(FuncInfo->getSGPRForEXECCopy()))
      FuncInfo->setSGPRForEXECCopy(UnusedLowSGPR);
  } else {
    // No SGPR spills to virtual VGPR lanes and hence there won't be any WWM
    // spills/copies. Reset the SGPR reserved for EXEC copy.
    FuncInfo->setSGPRForEXECCopy(AMDGPU::NoRegister);
  }

  SaveBlocks.clear();
  RestoreBlocks.clear();

  return MadeChange;
}

PreservedAnalyses
SILowerSGPRSpillsPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  auto *LIS = MFAM.getCachedResult<LiveIntervalsAnalysis>(MF);
  auto *Indexes = MFAM.getCachedResult<SlotIndexesAnalysis>(MF);
  MachineDominatorTree *MDT = &MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  SILowerSGPRSpills(LIS, Indexes, MDT).run(MF);
  return PreservedAnalyses::all();
}
