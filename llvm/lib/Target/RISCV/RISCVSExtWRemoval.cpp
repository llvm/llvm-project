//===-------------- RISCVSExtWRemoval.cpp - MI sext.w Removal -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass removes unneeded sext.w instructions at the MI level.
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-sextw-removal"

STATISTIC(NumRemovedSExtW, "Number of removed sign-extensions");

static cl::opt<bool> DisableSExtWRemoval("riscv-disable-sextw-removal",
                                         cl::desc("Disable removal of sext.w"),
                                         cl::init(false), cl::Hidden);
namespace {

class RISCVSExtWRemoval : public MachineFunctionPass {
public:
  static char ID;

  RISCVSExtWRemoval() : MachineFunctionPass(ID) {
    initializeRISCVSExtWRemovalPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return "RISCV sext.w Removal"; }
};

} // end anonymous namespace

char RISCVSExtWRemoval::ID = 0;
INITIALIZE_PASS(RISCVSExtWRemoval, DEBUG_TYPE, "RISCV sext.w Removal", false,
                false)

FunctionPass *llvm::createRISCVSExtWRemovalPass() {
  return new RISCVSExtWRemoval();
}

// This function returns true if the machine instruction always outputs a value
// where bits 63:32 match bit 31.
// TODO: Allocate a bit in TSFlags for the W instructions?
// TODO: Add other W instructions.
static bool isSignExtendingOpW(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::LUI:
  case RISCV::LW:
  case RISCV::ADDW:
  case RISCV::ADDIW:
  case RISCV::SUBW:
  case RISCV::MULW:
  case RISCV::SLLW:
  case RISCV::SLLIW:
  case RISCV::SRAW:
  case RISCV::SRAIW:
  case RISCV::SRLW:
  case RISCV::SRLIW:
  case RISCV::DIVW:
  case RISCV::DIVUW:
  case RISCV::REMW:
  case RISCV::REMUW:
  case RISCV::ROLW:
  case RISCV::RORW:
  case RISCV::RORIW:
  case RISCV::CLZW:
  case RISCV::CTZW:
  case RISCV::CPOPW:
  case RISCV::FCVT_W_H:
  case RISCV::FCVT_WU_H:
  case RISCV::FCVT_W_S:
  case RISCV::FCVT_WU_S:
  case RISCV::FCVT_W_D:
  case RISCV::FCVT_WU_D:
  case RISCV::FMV_X_W:
  // The following aren't W instructions, but are either sign extended from a
  // smaller size or put zeros in bits 63:31.
  case RISCV::LBU:
  case RISCV::LHU:
  case RISCV::LB:
  case RISCV::LH:
  case RISCV::SLT:
  case RISCV::SLTI:
  case RISCV::SLTU:
  case RISCV::SLTIU:
  case RISCV::SEXT_B:
  case RISCV::SEXT_H:
  case RISCV::ZEXT_H_RV64:
  case RISCV::FMV_X_H:
    return true;
  // shifting right sufficiently makes the value 32-bit sign-extended
  case RISCV::SRAI:
    return MI.getOperand(2).getImm() >= 32;
  case RISCV::SRLI:
    return MI.getOperand(2).getImm() > 32;
  // The LI pattern ADDI rd, X0, imm is sign extended.
  case RISCV::ADDI:
    return MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == RISCV::X0;
  // An ANDI with an 11 bit immediate will zero bits 63:11.
  case RISCV::ANDI:
    return isUInt<11>(MI.getOperand(2).getImm());
  // An ORI with an >11 bit immediate (negative 12-bit) will set bits 63:11.
  case RISCV::ORI:
    return !isUInt<11>(MI.getOperand(2).getImm());
  // Copying from X0 produces zero.
  case RISCV::COPY:
    return MI.getOperand(1).getReg() == RISCV::X0;
  }

  return false;
}

static bool isSignExtendedW(const MachineInstr &OrigMI,
                            MachineRegisterInfo &MRI) {

  SmallPtrSet<const MachineInstr *, 4> Visited;
  SmallVector<const MachineInstr *, 4> Worklist;

  Worklist.push_back(&OrigMI);

  while (!Worklist.empty()) {
    const MachineInstr *MI = Worklist.pop_back_val();

    // If we already visited this instruction, we don't need to check it again.
    if (!Visited.insert(MI).second)
      continue;

    // If this is a sign extending operation we don't need to look any further.
    if (isSignExtendingOpW(*MI))
      continue;

    // Is this an instruction that propagates sign extend.
    switch (MI->getOpcode()) {
    default:
      // Unknown opcode, give up.
      return false;
    case RISCV::COPY: {
      Register SrcReg = MI->getOperand(1).getReg();

      // TODO: Handle arguments and returns from calls?

      // If this is a copy from another register, check its source instruction.
      if (!SrcReg.isVirtual())
        return false;
      const MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);
      if (!SrcMI)
        return false;

      // Add SrcMI to the worklist.
      Worklist.push_back(SrcMI);
      break;
    }
    case RISCV::REM:
    case RISCV::ANDI:
    case RISCV::ORI:
    case RISCV::XORI: {
      // |Remainder| is always <= |Dividend|. If D is 32-bit, then so is R.
      // DIV doesn't work because of the edge case 0xf..f 8000 0000 / (long)-1
      // Logical operations use a sign extended 12-bit immediate. We just need
      // to check if the other operand is sign extended.
      Register SrcReg = MI->getOperand(1).getReg();
      if (!SrcReg.isVirtual())
        return false;
      const MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);
      if (!SrcMI)
        return false;

      // Add SrcMI to the worklist.
      Worklist.push_back(SrcMI);
      break;
    }
    case RISCV::REMU:
    case RISCV::AND:
    case RISCV::OR:
    case RISCV::XOR:
    case RISCV::ANDN:
    case RISCV::ORN:
    case RISCV::XNOR:
    case RISCV::MAX:
    case RISCV::MAXU:
    case RISCV::MIN:
    case RISCV::MINU:
    case RISCV::PHI: {
      // If all incoming values are sign-extended, the output of AND, OR, XOR,
      // MIN, MAX, or PHI is also sign-extended.

      // The input registers for PHI are operand 1, 3, ...
      // The input registers for others are operand 1 and 2.
      unsigned E = 3, D = 1;
      if (MI->getOpcode() == RISCV::PHI) {
        E = MI->getNumOperands();
        D = 2;
      }

      for (unsigned I = 1; I != E; I += D) {
        if (!MI->getOperand(I).isReg())
          return false;

        Register SrcReg = MI->getOperand(I).getReg();
        if (!SrcReg.isVirtual())
          return false;
        const MachineInstr *SrcMI = MRI.getVRegDef(SrcReg);
        if (!SrcMI)
          return false;

        // Add SrcMI to the worklist.
        Worklist.push_back(SrcMI);
      }

      break;
    }
    }
  }

  // If we get here, then every node we visited produces a sign extended value
  // or propagated sign extended values. So the result must be sign extended.
  return true;
}

bool RISCVSExtWRemoval::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || DisableSExtWRemoval)
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();

  if (!ST.is64Bit())
    return false;

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (auto I = MBB.begin(), IE = MBB.end(); I != IE;) {
      MachineInstr *MI = &*I++;

      // We're looking for the sext.w pattern ADDIW rd, rs1, 0.
      if (MI->getOpcode() != RISCV::ADDIW || !MI->getOperand(2).isImm() ||
          MI->getOperand(2).getImm() != 0 || !MI->getOperand(1).isReg())
        continue;

      // Input should be a virtual register.
      Register SrcReg = MI->getOperand(1).getReg();
      if (!SrcReg.isVirtual())
        continue;

      const MachineInstr &SrcMI = *MRI.getVRegDef(SrcReg);
      if (!isSignExtendedW(SrcMI, MRI))
        continue;

      Register DstReg = MI->getOperand(0).getReg();
      if (!MRI.constrainRegClass(SrcReg, MRI.getRegClass(DstReg)))
        continue;

      LLVM_DEBUG(dbgs() << "Removing redundant sign-extension\n");
      MRI.replaceRegWith(DstReg, SrcReg);
      MRI.clearKillFlags(SrcReg);
      MI->eraseFromParent();
      ++NumRemovedSExtW;
      MadeChange = true;
    }
  }

  return MadeChange;
}
