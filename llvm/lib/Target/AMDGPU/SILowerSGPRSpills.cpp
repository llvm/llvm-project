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

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-lower-sgpr-spills"

using MBBVector = SmallVector<MachineBasicBlock *, 4>;

namespace {

static cl::opt<bool> EnableSpillVGPRToAGPR(
  "amdgpu-spill-vgpr-to-agpr",
  cl::desc("Enable spilling VGPRs to AGPRs"),
  cl::ReallyHidden,
  cl::init(true));

class SILowerSGPRSpills : public MachineFunctionPass {
private:
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;

  // Save and Restore blocks of the current function. Typically there is a
  // single save block, unless Windows EH funclets are involved.
  MBBVector SaveBlocks;
  MBBVector RestoreBlocks;

public:
  static char ID;

  SILowerSGPRSpills() : MachineFunctionPass(ID) {}

  void calculateSaveRestoreBlocks(MachineFunction &MF);
  bool spillCalleeSavedRegs(MachineFunction &MF);

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SILowerSGPRSpills::ID = 0;

INITIALIZE_PASS_BEGIN(SILowerSGPRSpills, DEBUG_TYPE,
                      "SI lower SGPR spill instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(SILowerSGPRSpills, DEBUG_TYPE,
                    "SI lower SGPR spill instructions", false, false)

char &llvm::SILowerSGPRSpillsID = SILowerSGPRSpills::ID;

/// Insert restore code for the callee-saved registers used in the function.
static void insertCSRSaves(MachineBasicBlock &SaveBlock,
                           ArrayRef<CalleeSavedInfo> CSI,
                           LiveIntervals *LIS) {
  MachineFunction &MF = *SaveBlock.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  MachineBasicBlock::iterator I = SaveBlock.begin();
  if (!TFI->spillCalleeSavedRegisters(SaveBlock, I, CSI, TRI)) {
    const MachineRegisterInfo &MRI = MF.getRegInfo();

    for (const CalleeSavedInfo &CS : CSI) {
      // Insert the spill to the stack frame.
      MCRegister Reg = CS.getReg();

      MachineInstrSpan MIS(I, &SaveBlock);
      const TargetRegisterClass *RC =
        TRI->getMinimalPhysRegClass(Reg, MVT::i32);

      // If this value was already livein, we probably have a direct use of the
      // incoming register value, so don't kill at the spill point. This happens
      // since we pass some special inputs (workgroup IDs) in the callee saved
      // range.
      const bool IsLiveIn = MRI.isLiveIn(Reg);
      TII.storeRegToStackSlot(SaveBlock, I, Reg, !IsLiveIn, CS.getFrameIdx(),
                              RC, TRI);

      if (LIS) {
        assert(std::distance(MIS.begin(), I) == 1);
        MachineInstr &Inst = *std::prev(I);

        LIS->InsertMachineInstrInMaps(Inst);
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
  }
}

/// Insert restore code for the callee-saved registers used in the function.
static void insertCSRRestores(MachineBasicBlock &RestoreBlock,
                              MutableArrayRef<CalleeSavedInfo> CSI,
                              LiveIntervals *LIS) {
  MachineFunction &MF = *RestoreBlock.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  // Restore all registers immediately before the return and any
  // terminators that precede it.
  MachineBasicBlock::iterator I = RestoreBlock.getFirstTerminator();

  // FIXME: Just emit the readlane/writelane directly
  if (!TFI->restoreCalleeSavedRegisters(RestoreBlock, I, CSI, TRI)) {
    for (const CalleeSavedInfo &CI : reverse(CSI)) {
      unsigned Reg = CI.getReg();
      const TargetRegisterClass *RC =
        TRI->getMinimalPhysRegClass(Reg, MVT::i32);

      TII.loadRegFromStackSlot(RestoreBlock, I, Reg, CI.getFrameIdx(), RC, TRI);
      assert(I != RestoreBlock.begin() &&
             "loadRegFromStackSlot didn't insert any code!");
      // Insert in reverse order.  loadRegFromStackSlot can insert
      // multiple instructions.

      if (LIS) {
        MachineInstr &Inst = *std::prev(I);
        LIS->InsertMachineInstrInMaps(Inst);
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
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
  if (MFI.getSavePoint()) {
    SaveBlocks.push_back(MFI.getSavePoint());
    assert(MFI.getRestorePoint() && "Both restore and save must be set");
    MachineBasicBlock *RestoreBlock = MFI.getRestorePoint();
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

bool SILowerSGPRSpills::spillCalleeSavedRegs(MachineFunction &MF) {
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

        CSI.push_back(CalleeSavedInfo(Reg, JunkFI));
      }
    }

    if (!CSI.empty()) {
      for (MachineBasicBlock *SaveBlock : SaveBlocks)
        insertCSRSaves(*SaveBlock, CSI, LIS);

      // Add live ins to save blocks.
      assert(SaveBlocks.size() == 1 && "shrink wrapping not fully implemented");
      updateLiveness(MF, CSI);

      for (MachineBasicBlock *RestoreBlock : RestoreBlocks)
        insertCSRRestores(*RestoreBlock, CSI, LIS);
      return true;
    }
  }

  return false;
}

// Find lowest available VGPR and use it as VGPR reserved for SGPR spills.
static bool lowerShiftReservedVGPR(MachineFunction &MF,
                                   const GCNSubtarget &ST) {
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  const Register PreReservedVGPR = FuncInfo->VGPRReservedForSGPRSpill;
  // Early out if pre-reservation of a VGPR for SGPR spilling is disabled.
  if (!PreReservedVGPR)
    return false;

  // If there are no free lower VGPRs available, default to using the
  // pre-reserved register instead.
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  Register LowestAvailableVGPR =
      TRI->findUnusedRegister(MF.getRegInfo(), &AMDGPU::VGPR_32RegClass, MF);
  if (!LowestAvailableVGPR)
    LowestAvailableVGPR = PreReservedVGPR;

  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  // Create a stack object for a possible spill in the function prologue.
  // Note Non-CSR VGPR also need this as we may overwrite inactive lanes.
  Optional<int> FI = FrameInfo.CreateSpillStackObject(4, Align(4));

  // Find saved info about the pre-reserved register.
  const auto *ReservedVGPRInfoItr =
      llvm::find_if(FuncInfo->getSGPRSpillVGPRs(),
                    [PreReservedVGPR](const auto &SpillRegInfo) {
                      return SpillRegInfo.VGPR == PreReservedVGPR;
                    });

  assert(ReservedVGPRInfoItr != FuncInfo->getSGPRSpillVGPRs().end());
  auto Index =
      std::distance(FuncInfo->getSGPRSpillVGPRs().begin(), ReservedVGPRInfoItr);

  FuncInfo->setSGPRSpillVGPRs(LowestAvailableVGPR, FI, Index);

  for (MachineBasicBlock &MBB : MF) {
    assert(LowestAvailableVGPR.isValid() && "Did not find an available VGPR");
    MBB.addLiveIn(LowestAvailableVGPR);
    MBB.sortUniqueLiveIns();
  }

  return true;
}

bool SILowerSGPRSpills::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();

  VRM = getAnalysisIfAvailable<VirtRegMap>();

  assert(SaveBlocks.empty() && RestoreBlocks.empty());

  // First, expose any CSR SGPR spills. This is mostly the same as what PEI
  // does, but somewhat simpler.
  calculateSaveRestoreBlocks(MF);
  bool HasCSRs = spillCalleeSavedRegs(MF);

  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();

  if (!MFI.hasStackObjects() && !HasCSRs) {
    SaveBlocks.clear();
    RestoreBlocks.clear();
    if (FuncInfo->VGPRReservedForSGPRSpill) {
      // Free the reserved VGPR for later possible use by frame lowering.
      FuncInfo->removeVGPRForSGPRSpill(FuncInfo->VGPRReservedForSGPRSpill, MF);
      MRI.freezeReservedRegs(MF);
    }
    return false;
  }

  const bool SpillVGPRToAGPR = ST.hasMAIInsts() && FuncInfo->hasSpilledVGPRs()
    && EnableSpillVGPRToAGPR;

  bool MadeChange = false;

  const bool SpillToAGPR = EnableSpillVGPRToAGPR && ST.hasMAIInsts();
  std::unique_ptr<RegScavenger> RS;

  bool NewReservedRegs = false;

  // TODO: CSR VGPRs will never be spilled to AGPRs. These can probably be
  // handled as SpilledToReg in regular PrologEpilogInserter.
  const bool HasSGPRSpillToVGPR = TRI->spillSGPRToVGPR() &&
                                  (HasCSRs || FuncInfo->hasSpilledSGPRs());
  if (HasSGPRSpillToVGPR || SpillVGPRToAGPR) {
    // Process all SGPR spills before frame offsets are finalized. Ideally SGPRs
    // are spilled to VGPRs, in which case we can eliminate the stack usage.
    //
    // This operates under the assumption that only other SGPR spills are users
    // of the frame index.

    lowerShiftReservedVGPR(MF, ST);

    // To track the spill frame indices handled in this pass.
    BitVector SpillFIs(MFI.getObjectIndexEnd(), false);

    for (MachineBasicBlock &MBB : MF) {
      MachineBasicBlock::iterator Next;
      for (auto I = MBB.begin(), E = MBB.end(); I != E; I = Next) {
        MachineInstr &MI = *I;
        Next = std::next(I);

        if (SpillToAGPR && TII->isVGPRSpill(MI)) {
          // Try to eliminate stack used by VGPR spills before frame
          // finalization.
          unsigned FIOp = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                                     AMDGPU::OpName::vaddr);
          int FI = MI.getOperand(FIOp).getIndex();
          Register VReg =
              TII->getNamedOperand(MI, AMDGPU::OpName::vdata)->getReg();
          if (FuncInfo->allocateVGPRSpillToAGPR(MF, FI,
                                                TRI->isAGPR(MRI, VReg))) {
            NewReservedRegs = true;
            if (!RS)
              RS.reset(new RegScavenger());

            // FIXME: change to enterBasicBlockEnd()
            RS->enterBasicBlock(MBB);
            TRI->eliminateFrameIndex(MI, 0, FIOp, RS.get());
            SpillFIs.set(FI);
            continue;
          }
        }

        if (!TII->isSGPRSpill(MI) || !TRI->spillSGPRToVGPR())
          continue;

        int FI = TII->getNamedOperand(MI, AMDGPU::OpName::addr)->getIndex();
        assert(MFI.getStackID(FI) == TargetStackID::SGPRSpill);
        if (FuncInfo->allocateSGPRSpillToVGPR(MF, FI)) {
          NewReservedRegs = true;
          bool Spilled = TRI->eliminateSGPRToVGPRSpillFrameIndex(MI, FI, nullptr);
          (void)Spilled;
          assert(Spilled && "failed to spill SGPR to VGPR when allocated");
          SpillFIs.set(FI);
        }
      }
    }

    for (MachineBasicBlock &MBB : MF) {
      for (auto SSpill : FuncInfo->getSGPRSpillVGPRs())
        MBB.addLiveIn(SSpill.VGPR);

      for (MCPhysReg Reg : FuncInfo->getVGPRSpillAGPRs())
        MBB.addLiveIn(Reg);

      for (MCPhysReg Reg : FuncInfo->getAGPRSpillVGPRs())
        MBB.addLiveIn(Reg);

      MBB.sortUniqueLiveIns();

      // FIXME: The dead frame indices are replaced with a null register from
      // the debug value instructions. We should instead, update it with the
      // correct register value. But not sure the register value alone is
      // adequate to lower the DIExpression. It should be worked out later.
      for (MachineInstr &MI : MBB) {
        if (MI.isDebugValue() && MI.getOperand(0).isFI() &&
            SpillFIs[MI.getOperand(0).getIndex()]) {
          MI.getOperand(0).ChangeToRegister(Register(), false /*isDef*/);
          MI.getOperand(0).setIsDebug();
        }
      }
    }

    MadeChange = true;
  } else if (FuncInfo->VGPRReservedForSGPRSpill) {
    FuncInfo->removeVGPRForSGPRSpill(FuncInfo->VGPRReservedForSGPRSpill, MF);
  }

  SaveBlocks.clear();
  RestoreBlocks.clear();

  // Updated the reserved registers with any VGPRs added for SGPR spills.
  if (NewReservedRegs)
    MRI.freezeReservedRegs(MF);

  return MadeChange;
}
