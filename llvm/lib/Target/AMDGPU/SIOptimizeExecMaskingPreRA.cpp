//===-- SIOptimizeExecMaskingPreRA.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass performs exec mask handling peephole optimizations which needs
/// to be done before register allocation to reduce register pressure.
///
//===----------------------------------------------------------------------===//

#include "SIOptimizeExecMaskingPreRA.h"
#include "AMDGPU.h"
#include "AMDGPULaneMaskUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-optimize-exec-masking-pre-ra"

namespace {

class SIOptimizeExecMaskingPreRA {
private:
  const GCNSubtarget &ST;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  const AMDGPU::LaneMaskConstants &LMC;

  MCRegister ExecReg;

  bool optimizeElseBranch(MachineBasicBlock &MBB);

public:
  SIOptimizeExecMaskingPreRA(MachineFunction &MF, LiveIntervals *LIS)
      : ST(MF.getSubtarget<GCNSubtarget>()), TRI(ST.getRegisterInfo()),
        TII(ST.getInstrInfo()), MRI(&MF.getRegInfo()), LIS(LIS),
        LMC(AMDGPU::LaneMaskConstants::get(ST)) {}
  bool run(MachineFunction &MF);
};

class SIOptimizeExecMaskingPreRALegacy : public MachineFunctionPass {
public:
  static char ID;

  SIOptimizeExecMaskingPreRALegacy() : MachineFunctionPass(ID) {
    initializeSIOptimizeExecMaskingPreRALegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI optimize exec mask operations pre-RA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIOptimizeExecMaskingPreRALegacy, DEBUG_TYPE,
                      "SI optimize exec mask operations pre-RA", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(SIOptimizeExecMaskingPreRALegacy, DEBUG_TYPE,
                    "SI optimize exec mask operations pre-RA", false, false)

char SIOptimizeExecMaskingPreRALegacy::ID = 0;

char &llvm::SIOptimizeExecMaskingPreRAID = SIOptimizeExecMaskingPreRALegacy::ID;

FunctionPass *llvm::createSIOptimizeExecMaskingPreRAPass() {
  return new SIOptimizeExecMaskingPreRALegacy();
}

// Optimize sequence
//    %dst = S_OR_SAVEEXEC %src
//    ... instructions not modifying exec ...
//    %tmp = S_AND $exec, %dst
//    $exec = S_XOR_term $exec, %tmp
// =>
//    %dst = S_OR_SAVEEXEC %src
//    ... instructions not modifying exec ...
//    $exec = S_XOR_term $exec, %dst
//
// Clean up potentially unnecessary code added for safety during
// control flow lowering.
//
// Return whether any changes were made to MBB.
bool SIOptimizeExecMaskingPreRA::optimizeElseBranch(MachineBasicBlock &MBB) {
  if (MBB.empty())
    return false;

  // Check this is an else block.
  auto First = MBB.begin();
  MachineInstr &SaveExecMI = *First;
  if (SaveExecMI.getOpcode() != LMC.OrSaveExecOpc)
    return false;

  auto I = llvm::find_if(MBB.terminators(), [this](const MachineInstr &MI) {
    return MI.getOpcode() == LMC.XorTermOpc;
  });
  if (I == MBB.terminators().end())
    return false;

  MachineInstr &XorTermMI = *I;
  if (XorTermMI.getOperand(1).getReg() != Register(ExecReg))
    return false;

  Register SavedExecReg = SaveExecMI.getOperand(0).getReg();
  Register DstReg = XorTermMI.getOperand(2).getReg();

  // Find potentially unnecessary S_AND
  MachineInstr *AndExecMI = nullptr;
  I--;
  while (I != First && !AndExecMI) {
    if (I->getOpcode() == LMC.AndOpc && I->getOperand(0).getReg() == DstReg &&
        I->getOperand(1).getReg() == Register(ExecReg))
      AndExecMI = &*I;
    I--;
  }
  if (!AndExecMI)
    return false;

  // Check for exec modifying instructions.
  // Note: exec defs do not create live ranges beyond the
  // instruction so isDefBetween cannot be used.
  // Instead just check that the def segments are adjacent.
  SlotIndex StartIdx = LIS->getInstructionIndex(SaveExecMI);
  SlotIndex EndIdx = LIS->getInstructionIndex(*AndExecMI);
  for (MCRegUnit Unit : TRI->regunits(ExecReg)) {
    LiveRange &RegUnit = LIS->getRegUnit(Unit);
    if (RegUnit.find(StartIdx) != std::prev(RegUnit.find(EndIdx)))
      return false;
  }

  // Remove unnecessary S_AND
  LIS->removeInterval(SavedExecReg);
  LIS->removeInterval(DstReg);

  SaveExecMI.getOperand(0).setReg(DstReg);

  LIS->RemoveMachineInstrFromMaps(*AndExecMI);
  AndExecMI->eraseFromParent();

  LIS->createAndComputeVirtRegInterval(DstReg);

  return true;
}

PreservedAnalyses
SIOptimizeExecMaskingPreRAPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  auto &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SIOptimizeExecMaskingPreRA(MF, &LIS).run(MF);
  return PreservedAnalyses::all();
}

bool SIOptimizeExecMaskingPreRALegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  auto *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return SIOptimizeExecMaskingPreRA(MF, LIS).run(MF);
}

bool SIOptimizeExecMaskingPreRA::run(MachineFunction &MF) {
  ExecReg = MCRegister::from(LMC.ExecReg);

  DenseSet<Register> RecalcRegs({AMDGPU::EXEC_LO, AMDGPU::EXEC_HI});
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {

    if (optimizeElseBranch(MBB)) {
      RecalcRegs.insert(AMDGPU::SCC);
      Changed = true;
    }

    // Try to remove unneeded instructions before s_endpgm.
    if (MBB.succ_empty()) {
      if (MBB.empty())
        continue;

      // Skip this if the endpgm has any implicit uses, otherwise we would need
      // to be careful to update / remove them.
      // S_ENDPGM always has a single imm operand that is not used other than to
      // end up in the encoding
      MachineInstr &Term = MBB.back();
      if (Term.getOpcode() != AMDGPU::S_ENDPGM || Term.getNumOperands() != 1)
        continue;

      SmallVector<MachineBasicBlock*, 4> Blocks({&MBB});

      while (!Blocks.empty()) {
        auto *CurBB = Blocks.pop_back_val();
        auto I = CurBB->rbegin(), E = CurBB->rend();
        if (I != E) {
          if (I->isUnconditionalBranch() || I->getOpcode() == AMDGPU::S_ENDPGM)
            ++I;
          else if (I->isBranch())
            continue;
        }

        while (I != E) {
          if (I->isDebugInstr()) {
            I = std::next(I);
            continue;
          }

          if (I->mayStore() || I->isBarrier() || I->isCall() ||
              I->hasUnmodeledSideEffects() || I->hasOrderedMemoryRef())
            break;

          LLVM_DEBUG(dbgs()
                     << "Removing no effect instruction: " << *I << '\n');

          for (auto &Op : I->operands()) {
            if (Op.isReg())
              RecalcRegs.insert(Op.getReg());
          }

          auto Next = std::next(I);
          LIS->RemoveMachineInstrFromMaps(*I);
          I->eraseFromParent();
          I = Next;

          Changed = true;
        }

        if (I != E)
          continue;

        // Try to ascend predecessors.
        for (auto *Pred : CurBB->predecessors()) {
          if (Pred->succ_size() == 1)
            Blocks.push_back(Pred);
        }
      }
      continue;
    }

    // If the only user of a logical operation is move to exec, fold it now
    // to prevent forming of saveexec. I.e.:
    //
    //    %0:sreg_64 = COPY $exec
    //    %1:sreg_64 = S_AND_B64 %0:sreg_64, %2:sreg_64
    // =>
    //    %1 = S_AND_B64 $exec, %2:sreg_64
    unsigned ScanThreshold = 10;
    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E
         && ScanThreshold--; ++I) {
      // Continue scanning if this is not a full exec copy
      if (!(I->isFullCopy() && I->getOperand(1).getReg() == Register(ExecReg)))
        continue;

      Register SavedExec = I->getOperand(0).getReg();
      if (SavedExec.isVirtual() && MRI->hasOneNonDBGUse(SavedExec)) {
        MachineInstr *SingleExecUser = &*MRI->use_instr_nodbg_begin(SavedExec);
        int Idx = SingleExecUser->findRegisterUseOperandIdx(SavedExec,
                                                            /*TRI=*/nullptr);
        assert(Idx != -1);
        if (SingleExecUser->getParent() == I->getParent() &&
            !SingleExecUser->getOperand(Idx).isImplicit() &&
            TII->isOperandLegal(*SingleExecUser, Idx, &I->getOperand(1))) {
          LLVM_DEBUG(dbgs() << "Redundant EXEC COPY: " << *I << '\n');
          LIS->RemoveMachineInstrFromMaps(*I);
          I->eraseFromParent();
          MRI->replaceRegWith(SavedExec, ExecReg);
          LIS->removeInterval(SavedExec);
          Changed = true;
        }
      }
      break;
    }
  }

  if (Changed) {
    for (auto Reg : RecalcRegs) {
      if (Reg.isVirtual()) {
        LIS->removeInterval(Reg);
        if (!MRI->reg_empty(Reg))
          LIS->createAndComputeVirtRegInterval(Reg);
      } else {
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
  }

  return Changed;
}
