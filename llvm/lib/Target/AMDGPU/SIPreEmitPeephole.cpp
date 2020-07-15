//===-- SIPreEmitPeephole.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass performs the peephole optimizations before code emission.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-pre-emit-peephole"

namespace {

class SIPreEmitPeephole : public MachineFunctionPass {
private:
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;

  bool optimizeVccBranch(MachineInstr &MI) const;
  bool optimizeSetGPR(MachineInstr &First, MachineInstr &MI) const;

public:
  static char ID;

  SIPreEmitPeephole() : MachineFunctionPass(ID) {
    initializeSIPreEmitPeepholePass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // End anonymous namespace.

INITIALIZE_PASS(SIPreEmitPeephole, DEBUG_TYPE,
                "SI peephole optimizations", false, false)

char SIPreEmitPeephole::ID = 0;

char &llvm::SIPreEmitPeepholeID = SIPreEmitPeephole::ID;

bool SIPreEmitPeephole::optimizeVccBranch(MachineInstr &MI) const {
  // Match:
  // sreg = -1 or 0
  // vcc = S_AND_B64 exec, sreg or S_ANDN2_B64 exec, sreg
  // S_CBRANCH_VCC[N]Z
  // =>
  // S_CBRANCH_EXEC[N]Z
  // We end up with this pattern sometimes after basic block placement.
  // It happens while combining a block which assigns -1 or 0 to a saved mask
  // and another block which consumes that saved mask and then a branch.
  bool Changed = false;
  MachineBasicBlock &MBB = *MI.getParent();
  const GCNSubtarget &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
  const bool IsWave32 = ST.isWave32();
  const unsigned CondReg = TRI->getVCC();
  const unsigned ExecReg = IsWave32 ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  const unsigned And = IsWave32 ? AMDGPU::S_AND_B32 : AMDGPU::S_AND_B64;
  const unsigned AndN2 = IsWave32 ? AMDGPU::S_ANDN2_B32 : AMDGPU::S_ANDN2_B64;

  MachineBasicBlock::reverse_iterator A = MI.getReverseIterator(),
                                      E = MBB.rend();
  bool ReadsCond = false;
  unsigned Threshold = 5;
  for (++A; A != E; ++A) {
    if (!--Threshold)
      return false;
    if (A->modifiesRegister(ExecReg, TRI))
      return false;
    if (A->modifiesRegister(CondReg, TRI)) {
      if (!A->definesRegister(CondReg, TRI) ||
          (A->getOpcode() != And && A->getOpcode() != AndN2))
        return false;
      break;
    }
    ReadsCond |= A->readsRegister(CondReg, TRI);
  }
  if (A == E)
    return false;

  MachineOperand &Op1 = A->getOperand(1);
  MachineOperand &Op2 = A->getOperand(2);
  if (Op1.getReg() != ExecReg && Op2.isReg() && Op2.getReg() == ExecReg) {
    TII->commuteInstruction(*A);
    Changed = true;
  }
  if (Op1.getReg() != ExecReg)
    return Changed;
  if (Op2.isImm() && !(Op2.getImm() == -1 || Op2.getImm() == 0))
    return Changed;

  int64_t MaskValue = 0;
  Register SReg;
  if (Op2.isReg()) {
    SReg = Op2.getReg();
    auto M = std::next(A);
    bool ReadsSreg = false;
    for (; M != E; ++M) {
      if (M->definesRegister(SReg, TRI))
        break;
      if (M->modifiesRegister(SReg, TRI))
        return Changed;
      ReadsSreg |= M->readsRegister(SReg, TRI);
    }
    if (M == E || !M->isMoveImmediate() || !M->getOperand(1).isImm() ||
        (M->getOperand(1).getImm() != -1 && M->getOperand(1).getImm() != 0))
      return Changed;
    MaskValue = M->getOperand(1).getImm();
    // First if sreg is only used in the AND instruction fold the immediate
    // into into the AND.
    if (!ReadsSreg && Op2.isKill()) {
      A->getOperand(2).ChangeToImmediate(MaskValue);
      M->eraseFromParent();
    }
  } else if (Op2.isImm()) {
    MaskValue = Op2.getImm();
  } else {
    llvm_unreachable("Op2 must be register or immediate");
  }

  // Invert mask for s_andn2
  assert(MaskValue == 0 || MaskValue == -1);
  if (A->getOpcode() == AndN2)
    MaskValue = ~MaskValue;

  if (!ReadsCond && A->registerDefIsDead(AMDGPU::SCC) &&
      MI.killsRegister(CondReg, TRI))
    A->eraseFromParent();

  bool IsVCCZ = MI.getOpcode() == AMDGPU::S_CBRANCH_VCCZ;
  if (SReg == ExecReg) {
    // EXEC is updated directly
    if (IsVCCZ) {
      MI.eraseFromParent();
      return true;
    }
    MI.setDesc(TII->get(AMDGPU::S_BRANCH));
  } else if (IsVCCZ && MaskValue == 0) {
    // Will always branch
    // Remove all succesors shadowed by new unconditional branch
    MachineBasicBlock *Parent = MI.getParent();
    SmallVector<MachineInstr *, 4> ToRemove;
    bool Found = false;
    for (MachineInstr &Term : Parent->terminators()) {
      if (Found) {
        if (Term.isBranch())
          ToRemove.push_back(&Term);
      } else {
        Found = Term.isIdenticalTo(MI);
      }
    }
    assert(Found && "conditional branch is not terminator");
    for (auto BranchMI : ToRemove) {
      MachineOperand &Dst = BranchMI->getOperand(0);
      assert(Dst.isMBB() && "destination is not basic block");
      Parent->removeSuccessor(Dst.getMBB());
      BranchMI->eraseFromParent();
    }

    if (MachineBasicBlock *Succ = Parent->getFallThrough()) {
      Parent->removeSuccessor(Succ);
    }

    // Rewrite to unconditional branch
    MI.setDesc(TII->get(AMDGPU::S_BRANCH));
  } else if (!IsVCCZ && MaskValue == 0) {
    // Will never branch
    MachineOperand &Dst = MI.getOperand(0);
    assert(Dst.isMBB() && "destination is not basic block");
    MI.getParent()->removeSuccessor(Dst.getMBB());
    MI.eraseFromParent();
    return true;
  } else if (MaskValue == -1) {
    // Depends only on EXEC
    MI.setDesc(
        TII->get(IsVCCZ ? AMDGPU::S_CBRANCH_EXECZ : AMDGPU::S_CBRANCH_EXECNZ));
  }

  MI.RemoveOperand(MI.findRegisterUseOperandIdx(CondReg, false /*Kill*/, TRI));
  MI.addImplicitDefUseOperands(*MBB.getParent());

  return true;
}

bool SIPreEmitPeephole::optimizeSetGPR(MachineInstr &First,
                                       MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  const MachineFunction &MF = *MBB.getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  Register IdxReg = Idx->isReg() ? Idx->getReg() : Register();
  SmallVector<MachineInstr *, 4> ToRemove;
  bool IdxOn = true;

  if (!MI.isIdenticalTo(First))
    return false;

  // Scan back to find an identical S_SET_GPR_IDX_ON
  for (MachineBasicBlock::iterator I = std::next(First.getIterator()),
       E = MI.getIterator(); I != E; ++I) {
    switch (I->getOpcode()) {
    case AMDGPU::S_SET_GPR_IDX_MODE:
      return false;
    case AMDGPU::S_SET_GPR_IDX_OFF:
      IdxOn = false;
      ToRemove.push_back(&*I);
      break;
    default:
      if (I->modifiesRegister(AMDGPU::M0, TRI))
        return false;
      if (IdxReg && I->modifiesRegister(IdxReg, TRI))
        return false;
      if (llvm::any_of(I->operands(),
                       [&MRI, this](const MachineOperand &MO) {
                         return MO.isReg() &&
                                TRI->isVectorRegister(MRI, MO.getReg());
                       })) {
        // The only exception allowed here is another indirect vector move
        // with the same mode.
        if (!IdxOn ||
            !((I->getOpcode() == AMDGPU::V_MOV_B32_e32 &&
               I->hasRegisterImplicitUseOperand(AMDGPU::M0)) ||
              I->getOpcode() == AMDGPU::V_MOV_B32_indirect))
          return false;
      }
    }
  }

  MI.eraseFromParent();
  for (MachineInstr *RI : ToRemove)
    RI->eraseFromParent();
  return true;
}

bool SIPreEmitPeephole::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MachineBasicBlock *EmptyMBBAtEnd = nullptr;
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::iterator MBBE = MBB.getFirstTerminator();
    if (MBBE != MBB.end()) {
      MachineInstr &MI = *MBBE;
      switch (MI.getOpcode()) {
      case AMDGPU::S_CBRANCH_VCCZ:
      case AMDGPU::S_CBRANCH_VCCNZ:
        Changed |= optimizeVccBranch(MI);
        continue;
      case AMDGPU::SI_RETURN_TO_EPILOG:
        // FIXME: This is not an optimization and should be
        // moved somewhere else.
        assert(!MF.getInfo<SIMachineFunctionInfo>()->returnsVoid());

        // Graphics shaders returning non-void shouldn't contain S_ENDPGM,
        // because external bytecode will be appended at the end.
        if (&MBB != &MF.back() || &MI != &MBB.back()) {
          // SI_RETURN_TO_EPILOG is not the last instruction. Add an empty block
          // at the end and jump there.
          if (!EmptyMBBAtEnd) {
            EmptyMBBAtEnd = MF.CreateMachineBasicBlock();
            MF.insert(MF.end(), EmptyMBBAtEnd);
          }

          MBB.addSuccessor(EmptyMBBAtEnd);
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(AMDGPU::S_BRANCH))
              .addMBB(EmptyMBBAtEnd);
          MI.eraseFromParent();
          MBBE = MBB.getFirstTerminator();
        }
        break;
      default:
        break;
      }
    }

    if (!ST.hasVGPRIndexMode())
      continue;

    MachineInstr *SetGPRMI = nullptr;
    const unsigned Threshold = 20;
    unsigned Count = 0;
    // Scan the block for two S_SET_GPR_IDX_ON instructions to see if a
    // second is not needed. Do expensive checks in the optimizeSetGPR()
    // and limit the distance to 20 instructions for compile time purposes.
    for (MachineBasicBlock::iterator MBBI = MBB.begin(); MBBI != MBBE; ) {
      MachineInstr &MI = *MBBI;
      ++MBBI;

      if (Count == Threshold)
        SetGPRMI = nullptr;
      else
        ++Count;

      if (MI.getOpcode() != AMDGPU::S_SET_GPR_IDX_ON)
        continue;

      Count = 0;
      if (!SetGPRMI) {
        SetGPRMI = &MI;
        continue;
      }

      if (optimizeSetGPR(*SetGPRMI, MI))
        Changed = true;
      else
        SetGPRMI = &MI;
    }
  }

  return Changed;
}
