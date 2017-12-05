//===-- SIOptimizeExecMaskingPreRA.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass removes redundant S_OR_B64 instructions enabling lanes in
/// the exec. If two SI_END_CF (lowered as S_OR_B64) come together without any
/// vector instructions between them we can only keep outer SI_END_CF, given
/// that CFG is structured and exec bits of the outer end statement are always
/// not less than exec bit of the inner one.
///
/// This needs to be done before the RA to eliminate saved exec bits registers
/// but after register coalescer to have no vector registers copies in between
/// of different end cf statements.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "si-optimize-exec-masking-pre-ra"

namespace {

class SIOptimizeExecMaskingPreRA : public MachineFunctionPass {
public:
  static char ID;

public:
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

static bool isEndCF(const MachineInstr& MI, const SIRegisterInfo* TRI) {
  return MI.getOpcode() == AMDGPU::S_OR_B64 &&
         MI.modifiesRegister(AMDGPU::EXEC, TRI);
}

static bool isFullExecCopy(const MachineInstr& MI) {
  return MI.isFullCopy() && MI.getOperand(1).getReg() == AMDGPU::EXEC;
}

static unsigned getOrNonExecReg(const MachineInstr &MI,
                                const SIInstrInfo &TII) {
  auto Op = TII.getNamedOperand(MI, AMDGPU::OpName::src1);
  if (Op->isReg() && Op->getReg() != AMDGPU::EXEC)
     return Op->getReg();
  Op = TII.getNamedOperand(MI, AMDGPU::OpName::src0);
  if (Op->isReg() && Op->getReg() != AMDGPU::EXEC)
     return Op->getReg();
  return AMDGPU::NoRegister;
}

static MachineInstr* getOrExecSource(const MachineInstr &MI,
                                     const SIInstrInfo &TII,
                                     const MachineRegisterInfo &MRI) {
  auto SavedExec = getOrNonExecReg(MI, TII);
  if (SavedExec == AMDGPU::NoRegister)
    return nullptr;
  auto SaveExecInst = MRI.getUniqueVRegDef(SavedExec);
  if (!SaveExecInst || !isFullExecCopy(*SaveExecInst))
    return nullptr;
  return SaveExecInst;
}

bool SIOptimizeExecMaskingPreRA::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  LiveIntervals *LIS = &getAnalysis<LiveIntervals>();
  DenseSet<unsigned> RecalcRegs({AMDGPU::EXEC_LO, AMDGPU::EXEC_HI});
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {

    // Try to remove unneeded instructions before s_endpgm.
    if (MBB.succ_empty()) {
      if (MBB.empty() || MBB.back().getOpcode() != AMDGPU::S_ENDPGM)
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
          if (I->isDebugValue()) {
            I = std::next(I);
            continue;
          }

          if (I->mayStore() || I->isBarrier() || I->isCall() ||
              I->hasUnmodeledSideEffects() || I->hasOrderedMemoryRef())
            break;

          DEBUG(dbgs() << "Removing no effect instruction: " << *I << '\n');

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

    // Try to collapse adjacent endifs.
    auto Lead = MBB.begin(), E = MBB.end();
    if (MBB.succ_size() != 1 || Lead == E || !isEndCF(*Lead, TRI))
      continue;

    const MachineBasicBlock* Succ = *MBB.succ_begin();
    if (!MBB.isLayoutSuccessor(Succ))
      continue;

    auto I = std::next(Lead);

    for ( ; I != E; ++I)
      if (!TII->isSALU(*I) || I->readsRegister(AMDGPU::EXEC, TRI))
        break;

    if (I != E)
      continue;

    const auto NextLead = Succ->begin();
    if (NextLead == Succ->end() || !isEndCF(*NextLead, TRI) ||
        !getOrExecSource(*NextLead, *TII, MRI))
      continue;

    DEBUG(dbgs() << "Redundant EXEC = S_OR_B64 found: " << *Lead << '\n');

    auto SaveExec = getOrExecSource(*Lead, *TII, MRI);
    unsigned SaveExecReg = getOrNonExecReg(*Lead, *TII);
    for (auto &Op : Lead->operands()) {
      if (Op.isReg())
        RecalcRegs.insert(Op.getReg());
    }

    LIS->RemoveMachineInstrFromMaps(*Lead);
    Lead->eraseFromParent();
    if (SaveExecReg) {
      LIS->removeInterval(SaveExecReg);
      LIS->createAndComputeVirtRegInterval(SaveExecReg);
    }

    Changed = true;

    // If the only use of saved exec in the removed instruction is S_AND_B64
    // fold the copy now.
    if (!SaveExec || !SaveExec->isFullCopy())
      continue;

    unsigned SavedExec = SaveExec->getOperand(0).getReg();
    bool SafeToReplace = true;
    for (auto& U : MRI.use_nodbg_instructions(SavedExec)) {
      if (U.getParent() != SaveExec->getParent()) {
        SafeToReplace = false;
        break;
      }

      DEBUG(dbgs() << "Redundant EXEC COPY: " << *SaveExec << '\n');
    }

    if (SafeToReplace) {
      LIS->RemoveMachineInstrFromMaps(*SaveExec);
      SaveExec->eraseFromParent();
      MRI.replaceRegWith(SavedExec, AMDGPU::EXEC);
      LIS->removeInterval(SavedExec);
    }
  }

  if (Changed) {
    for (auto Reg : RecalcRegs) {
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        LIS->removeInterval(Reg);
        if (!MRI.reg_empty(Reg))
          LIS->createAndComputeVirtRegInterval(Reg);
      } else {
        for (MCRegUnitIterator U(Reg, TRI); U.isValid(); ++U)
          LIS->removeRegUnit(*U);
      }
    }
  }

  return Changed;
}
