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

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-optimize-exec-masking-pre-ra"

namespace {

class SIOptimizeExecMaskingPreRA : public MachineFunctionPass {
private:
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  MachineRegisterInfo *MRI;

public:
  static char ID;

  SIOptimizeExecMaskingPreRA() : MachineFunctionPass(ID) {
    initializeSIOptimizeExecMaskingPreRAPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI optimize exec mask operations pre-RA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIOptimizeExecMaskingPreRA, DEBUG_TYPE,
                      "SI optimize exec mask operations pre-RA", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(SIOptimizeExecMaskingPreRA, DEBUG_TYPE,
                    "SI optimize exec mask operations pre-RA", false, false)

char SIOptimizeExecMaskingPreRA::ID = 0;

char &llvm::SIOptimizeExecMaskingPreRAID = SIOptimizeExecMaskingPreRA::ID;

FunctionPass *llvm::createSIOptimizeExecMaskingPreRAPass() {
  return new SIOptimizeExecMaskingPreRA();
}

static bool isFullExecCopy(const MachineInstr& MI, const GCNSubtarget& ST) {
  unsigned Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;

  if (MI.isCopy() && MI.getOperand(1).getReg() == Exec) {
    assert(MI.isFullCopy());
    return true;
  }

  return false;
}

// Optimize sequence
//    %sel = V_CNDMASK_B32_e64 0, 1, %cc
//    %cmp = V_CMP_NE_U32 1, %1
//    $vcc = S_AND_B64 $exec, %cmp
//    S_CBRANCH_VCC[N]Z
// =>
//    $vcc = S_ANDN2_B64 $exec, %cc
//    S_CBRANCH_VCC[N]Z
//
// It is the negation pattern inserted by DAGCombiner::visitBRCOND() in the
// rebuildSetCC(). We start with S_CBRANCH to avoid exhaustive search, but
// only 3 first instructions are really needed. S_AND_B64 with exec is a
// required part of the pattern since V_CNDMASK_B32 writes zeroes for inactive
// lanes.
//
// Returns %cc register on success.
static unsigned optimizeVcndVcmpPair(MachineBasicBlock &MBB,
                                     const GCNSubtarget &ST,
                                     MachineRegisterInfo &MRI,
                                     LiveIntervals *LIS) {
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool Wave32 = ST.isWave32();
  const unsigned AndOpc = Wave32 ? AMDGPU::S_AND_B32 : AMDGPU::S_AND_B64;
  const unsigned Andn2Opc = Wave32 ? AMDGPU::S_ANDN2_B32 : AMDGPU::S_ANDN2_B64;
  const unsigned CondReg = Wave32 ? AMDGPU::VCC_LO : AMDGPU::VCC;
  const unsigned ExecReg = Wave32 ? AMDGPU::EXEC_LO : AMDGPU::EXEC;

  auto I = llvm::find_if(MBB.terminators(), [](const MachineInstr &MI) {
                           unsigned Opc = MI.getOpcode();
                           return Opc == AMDGPU::S_CBRANCH_VCCZ ||
                                  Opc == AMDGPU::S_CBRANCH_VCCNZ; });
  if (I == MBB.terminators().end())
    return AMDGPU::NoRegister;

  auto *And = TRI->findReachingDef(CondReg, AMDGPU::NoSubRegister,
                                   *I, MRI, LIS);
  if (!And || And->getOpcode() != AndOpc ||
      !And->getOperand(1).isReg() || !And->getOperand(2).isReg())
    return AMDGPU::NoRegister;

  MachineOperand *AndCC = &And->getOperand(1);
  Register CmpReg = AndCC->getReg();
  unsigned CmpSubReg = AndCC->getSubReg();
  if (CmpReg == ExecReg) {
    AndCC = &And->getOperand(2);
    CmpReg = AndCC->getReg();
    CmpSubReg = AndCC->getSubReg();
  } else if (And->getOperand(2).getReg() != ExecReg) {
    return AMDGPU::NoRegister;
  }

  auto *Cmp = TRI->findReachingDef(CmpReg, CmpSubReg, *And, MRI, LIS);
  if (!Cmp || !(Cmp->getOpcode() == AMDGPU::V_CMP_NE_U32_e32 ||
                Cmp->getOpcode() == AMDGPU::V_CMP_NE_U32_e64) ||
      Cmp->getParent() != And->getParent())
    return AMDGPU::NoRegister;

  MachineOperand *Op1 = TII->getNamedOperand(*Cmp, AMDGPU::OpName::src0);
  MachineOperand *Op2 = TII->getNamedOperand(*Cmp, AMDGPU::OpName::src1);
  if (Op1->isImm() && Op2->isReg())
    std::swap(Op1, Op2);
  if (!Op1->isReg() || !Op2->isImm() || Op2->getImm() != 1)
    return AMDGPU::NoRegister;

  Register SelReg = Op1->getReg();
  auto *Sel = TRI->findReachingDef(SelReg, Op1->getSubReg(), *Cmp, MRI, LIS);
  if (!Sel || Sel->getOpcode() != AMDGPU::V_CNDMASK_B32_e64)
    return AMDGPU::NoRegister;

  if (TII->hasModifiersSet(*Sel, AMDGPU::OpName::src0_modifiers) ||
      TII->hasModifiersSet(*Sel, AMDGPU::OpName::src1_modifiers))
    return AMDGPU::NoRegister;

  Op1 = TII->getNamedOperand(*Sel, AMDGPU::OpName::src0);
  Op2 = TII->getNamedOperand(*Sel, AMDGPU::OpName::src1);
  MachineOperand *CC = TII->getNamedOperand(*Sel, AMDGPU::OpName::src2);
  if (!Op1->isImm() || !Op2->isImm() || !CC->isReg() ||
      Op1->getImm() != 0 || Op2->getImm() != 1)
    return AMDGPU::NoRegister;

  LLVM_DEBUG(dbgs() << "Folding sequence:\n\t" << *Sel << '\t' << *Cmp << '\t'
                    << *And);

  Register CCReg = CC->getReg();
  LIS->RemoveMachineInstrFromMaps(*And);
  MachineInstr *Andn2 =
      BuildMI(MBB, *And, And->getDebugLoc(), TII->get(Andn2Opc),
              And->getOperand(0).getReg())
          .addReg(ExecReg)
          .addReg(CCReg, getUndefRegState(CC->isUndef()), CC->getSubReg());
  And->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*Andn2);

  LLVM_DEBUG(dbgs() << "=>\n\t" << *Andn2 << '\n');

  // Try to remove compare. Cmp value should not used in between of cmp
  // and s_and_b64 if VCC or just unused if any other register.
  if ((Register::isVirtualRegister(CmpReg) && MRI.use_nodbg_empty(CmpReg)) ||
      (CmpReg == CondReg &&
       std::none_of(std::next(Cmp->getIterator()), Andn2->getIterator(),
                    [&](const MachineInstr &MI) {
                      return MI.readsRegister(CondReg, TRI);
                    }))) {
    LLVM_DEBUG(dbgs() << "Erasing: " << *Cmp << '\n');

    LIS->RemoveMachineInstrFromMaps(*Cmp);
    Cmp->eraseFromParent();

    // Try to remove v_cndmask_b32.
    if (Register::isVirtualRegister(SelReg) && MRI.use_nodbg_empty(SelReg)) {
      LLVM_DEBUG(dbgs() << "Erasing: " << *Sel << '\n');

      LIS->RemoveMachineInstrFromMaps(*Sel);
      Sel->eraseFromParent();
    }
  }

  return CCReg;
}

bool SIOptimizeExecMaskingPreRA::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  MachineRegisterInfo &MRI = MF.getRegInfo();
  LiveIntervals *LIS = &getAnalysis<LiveIntervals>();
  DenseSet<unsigned> RecalcRegs({AMDGPU::EXEC_LO, AMDGPU::EXEC_HI});
  unsigned Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {

    if (unsigned Reg = optimizeVcndVcmpPair(MBB, ST, MRI, LIS)) {
      RecalcRegs.insert(Reg);
      RecalcRegs.insert(AMDGPU::VCC_LO);
      RecalcRegs.insert(AMDGPU::VCC_HI);
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
        auto CurBB = Blocks.pop_back_val();
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
    // to prevent forming of saveexec. I.e:
    //
    //    %0:sreg_64 = COPY $exec
    //    %1:sreg_64 = S_AND_B64 %0:sreg_64, %2:sreg_64
    // =>
    //    %1 = S_AND_B64 $exec, %2:sreg_64
    unsigned ScanThreshold = 10;
    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E
         && ScanThreshold--; ++I) {
      if (!isFullExecCopy(*I, ST))
        continue;

      Register SavedExec = I->getOperand(0).getReg();
      if (SavedExec.isVirtual() && MRI.hasOneNonDBGUse(SavedExec) &&
          MRI.use_instr_nodbg_begin(SavedExec)->getParent() == I->getParent()) {
        LLVM_DEBUG(dbgs() << "Redundant EXEC COPY: " << *I << '\n');
        LIS->RemoveMachineInstrFromMaps(*I);
        I->eraseFromParent();
        MRI.replaceRegWith(SavedExec, Exec);
        LIS->removeInterval(SavedExec);
        Changed = true;
      }
      break;
    }
  }

  if (Changed) {
    for (auto Reg : RecalcRegs) {
      if (Register::isVirtualRegister(Reg)) {
        LIS->removeInterval(Reg);
        if (!MRI.reg_empty(Reg))
          LIS->createAndComputeVirtRegInterval(Reg);
      } else {
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
  }

  return Changed;
}
