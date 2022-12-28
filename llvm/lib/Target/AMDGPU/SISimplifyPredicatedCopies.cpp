//===-- SISimplifyPredicatedCopies.cpp - Simplify Copies after regalloc
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Simplify the predicated COPY (PRED_COPY) instructions for various register
/// classes. AMDGPU vector register copies have a predicated dependency with
/// EXEC register and should be marked exec as an implicit operand post-RA. The
/// scalar registers don't carry any such dependency and hence the regular COPY
/// opcode can be used. AMDGPU by default uses PRED_COPY opcode right from the
/// instruction selection and this pass would simplify the COPY opcode and the
/// implicit operand field as mentioned above. This pass also implements the
/// EXEC MASK manipulation around the whole wave vector register copies by
/// turning all bits of exec to one before the copy and then restore it
/// immediately afterwards.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-simplify-predicated-copies"

namespace {

class SISimplifyPredicatedCopies : public MachineFunctionPass {
public:
  static char ID;

  SISimplifyPredicatedCopies() : MachineFunctionPass(ID) {
    initializeSISimplifyPredicatedCopiesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Simplify Predicated Copies";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool isWWMCopy(const MachineInstr &MI);
  bool isSCCLiveAtMI(const MachineInstr &MI);

  LiveIntervals *LIS;
  SlotIndexes *Indexes;
  const SIRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  SIMachineFunctionInfo *MFI;
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SISimplifyPredicatedCopies, DEBUG_TYPE,
                      "SI Simplify Predicated Copies", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(SISimplifyPredicatedCopies, DEBUG_TYPE,
                    "SI Simplify Predicated Copies", false, false)

char SISimplifyPredicatedCopies::ID = 0;

char &llvm::SISimplifyPredicatedCopiesID = SISimplifyPredicatedCopies::ID;

// Returns true if \p MI is a whole-wave copy instruction. Iterate
// recursively skipping the intermediate copies if it maps to any
// whole-wave operation.
bool SISimplifyPredicatedCopies::isWWMCopy(const MachineInstr &MI) {
  Register SrcReg = MI.getOperand(1).getReg();

  if (MFI->checkFlag(SrcReg, AMDGPU::VirtRegFlag::WWM_REG))
    return true;

  if (SrcReg.isPhysical())
    return false;

  // Look recursively skipping intermediate copies.
  const MachineInstr *DefMI = MRI->getUniqueVRegDef(SrcReg);
  if (!DefMI || !DefMI->isCopy())
    return false;

  return isWWMCopy(*DefMI);
}

bool SISimplifyPredicatedCopies::isSCCLiveAtMI(const MachineInstr &MI) {
  // We can't determine the liveness info if LIS isn't available. Early return
  // in that case and always assume SCC is live.
  if (!LIS)
    return true;

  LiveRange &LR =
      LIS->getRegUnit(*MCRegUnitIterator(MCRegister::from(AMDGPU::SCC), TRI));
  SlotIndex Idx = LIS->getInstructionIndex(MI);
  return LR.liveAt(Idx);
}

bool SISimplifyPredicatedCopies::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();

  MFI = MF.getInfo<SIMachineFunctionInfo>();
  LIS = getAnalysisIfAvailable<LiveIntervals>();
  Indexes = getAnalysisIfAvailable<SlotIndexes>();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      switch (Opcode) {
      case AMDGPU::COPY:
      case AMDGPU::PRED_COPY:
        if (!TII->isVGPRCopy(MI) &&
            TRI->isSGPRReg(*MRI, MI.getOperand(1).getReg())) {
          // For PRED_COPY with SGPR regclass, change the opcode back to the
          // regular COPY.
          if (Opcode == AMDGPU::PRED_COPY) {
            LLVM_DEBUG(dbgs() << MI << " to use COPY opcode");
            MI.setDesc(TII->get(AMDGPU::COPY));
            Changed = true;
          }
        } else {
          if (TII->isVGPRCopy(MI) &&
              !TRI->isSGPRReg(*MRI, MI.getOperand(1).getReg()) &&
              MI.getOperand(0).getReg().isVirtual() && isWWMCopy(MI)) {
            // For WWM vector copies, manipulate the exec mask around the copy
            // instruction.
            DebugLoc DL = MI.getDebugLoc();
            MachineBasicBlock::iterator InsertPt = MI.getIterator();
            Register RegForExecCopy = MFI->getSGPRForEXECCopy();
            TII->insertScratchExecCopy(MF, MBB, InsertPt, DL, RegForExecCopy,
                                       isSCCLiveAtMI(MI), Indexes);
            TII->restoreExec(MF, MBB, ++InsertPt, DL, RegForExecCopy, Indexes);
            LLVM_DEBUG(dbgs() << "WWM copy manipulation for " << MI);
          }

          // For vector registers, add implicit exec use.
          if (!MI.readsRegister(AMDGPU::EXEC, TRI)) {
            MI.addOperand(MF,
                          MachineOperand::CreateReg(AMDGPU::EXEC, false, true));
            LLVM_DEBUG(dbgs() << "Add exec use to " << MI);
            Changed = true;
          }
        }
        break;
      default:
        break;
      }
    }
  }

  return Changed;
}
