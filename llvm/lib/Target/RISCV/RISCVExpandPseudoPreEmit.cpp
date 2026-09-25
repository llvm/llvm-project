//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains one of the four passes that expand pseudo instructions
// into target instructions. This pass is run very late, but before atomic
// instructions are expanded.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVExpandPseudoBase.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define RISCV_EXPAND_PSEUDO_PRE_EMIT_NAME                                      \
  "RISC-V Pseudo Instruction Expansion - Pre-Emit"

namespace {

class RISCVExpandPseudoPreEmitImpl final : public RISCVExpandPseudoImplBase {
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI) const override;

  bool expandCCOp(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NextMBBI) const;

  bool expandCCOpToCMov(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI) const;

  bool expandVMSET_VMCLR(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI,
                         unsigned Opcode) const;

  bool expandMV_FPR16INX(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI) const;

  bool expandMV_FPR32INX(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI) const;

  bool expandRV32ZdinxStore(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI) const;

  bool expandRV32ZdinxLoad(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI) const;

  bool
  expandPseudoReadVLENBViaVSETVLIX0(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI) const;

  bool expandPseudoClearFPR64(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI) const;
};

class RISCVExpandPseudoPreEmitLegacy : public MachineFunctionPass {
public:
  static char ID;

  RISCVExpandPseudoPreEmitLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    return RISCVExpandPseudoPreEmitImpl().run(MF);
  }

  StringRef getPassName() const override {
    return RISCV_EXPAND_PSEUDO_PRE_EMIT_NAME;
  }
};

} // anonymous namespace

bool RISCVExpandPseudoPreEmitImpl::expandMI(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) const {
  // RISCVInstrInfo::getInstSizeInBytes expects that the total size of the
  // expanded instructions for each pseudo is correct in the Size field of the
  // tablegen definition for the pseudo.
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoMV_FPR16INX:
    return expandMV_FPR16INX(MBB, MBBI);
  case RISCV::PseudoMV_FPR32INX:
    return expandMV_FPR32INX(MBB, MBBI);
  case RISCV::PseudoRV32ZdinxSD:
    return expandRV32ZdinxStore(MBB, MBBI);
  case RISCV::PseudoRV32ZdinxLD:
    return expandRV32ZdinxLoad(MBB, MBBI);
  case RISCV::PseudoCCMOVGPRNoX0:
  case RISCV::PseudoCCMOVGPR:
  case RISCV::PseudoCCADD:
  case RISCV::PseudoCCSUB:
  case RISCV::PseudoCCAND:
  case RISCV::PseudoCCOR:
  case RISCV::PseudoCCXOR:
  case RISCV::PseudoCCMAX:
  case RISCV::PseudoCCMAXU:
  case RISCV::PseudoCCMIN:
  case RISCV::PseudoCCMINU:
  case RISCV::PseudoCCMUL:
  case RISCV::PseudoCCLUI:
  case RISCV::PseudoCCQC_E_LB:
  case RISCV::PseudoCCQC_E_LH:
  case RISCV::PseudoCCQC_E_LW:
  case RISCV::PseudoCCQC_E_LHU:
  case RISCV::PseudoCCQC_E_LBU:
  case RISCV::PseudoCCLB:
  case RISCV::PseudoCCLH:
  case RISCV::PseudoCCLW:
  case RISCV::PseudoCCLHU:
  case RISCV::PseudoCCLBU:
  case RISCV::PseudoCCLWU:
  case RISCV::PseudoCCLD:
  case RISCV::PseudoCCQC_LI:
  case RISCV::PseudoCCQC_E_LI:
  case RISCV::PseudoCCADDW:
  case RISCV::PseudoCCSUBW:
  case RISCV::PseudoCCSLL:
  case RISCV::PseudoCCSRL:
  case RISCV::PseudoCCSRA:
  case RISCV::PseudoCCADDI:
  case RISCV::PseudoCCSLLI:
  case RISCV::PseudoCCSRLI:
  case RISCV::PseudoCCSRAI:
  case RISCV::PseudoCCANDI:
  case RISCV::PseudoCCORI:
  case RISCV::PseudoCCXORI:
  case RISCV::PseudoCCSLLW:
  case RISCV::PseudoCCSRLW:
  case RISCV::PseudoCCSRAW:
  case RISCV::PseudoCCADDIW:
  case RISCV::PseudoCCSLLIW:
  case RISCV::PseudoCCSRLIW:
  case RISCV::PseudoCCSRAIW:
  case RISCV::PseudoCCANDN:
  case RISCV::PseudoCCORN:
  case RISCV::PseudoCCXNOR:
  case RISCV::PseudoCCNDS_BFOS:
  case RISCV::PseudoCCNDS_BFOZ:
    return expandCCOp(MBB, MBBI, NextMBBI);
  case RISCV::PseudoVMCLR_M_B1:
  case RISCV::PseudoVMCLR_M_B2:
  case RISCV::PseudoVMCLR_M_B4:
  case RISCV::PseudoVMCLR_M_B8:
  case RISCV::PseudoVMCLR_M_B16:
  case RISCV::PseudoVMCLR_M_B32:
  case RISCV::PseudoVMCLR_M_B64:
    // vmclr.m vd => vmxor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, RISCV::VMXOR_MM);
  case RISCV::PseudoVMSET_M_B1:
  case RISCV::PseudoVMSET_M_B2:
  case RISCV::PseudoVMSET_M_B4:
  case RISCV::PseudoVMSET_M_B8:
  case RISCV::PseudoVMSET_M_B16:
  case RISCV::PseudoVMSET_M_B32:
  case RISCV::PseudoVMSET_M_B64:
    // vmset.m vd => vmxnor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, RISCV::VMXNOR_MM);
  case RISCV::PseudoReadVLENBViaVSETVLIX0:
    return expandPseudoReadVLENBViaVSETVLIX0(MBB, MBBI);
  case RISCV::PseudoClearFPR64:
    return expandPseudoClearFPR64(MBB, MBBI);
  }

  return false;
}

bool RISCVExpandPseudoPreEmitImpl::expandCCOp(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) const {
  // First try expanding to a Conditional Move rather than a branch+mv
  if (expandCCOpToCMov(MBB, MBBI))
    return true;

  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  MachineBasicBlock *TrueBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *MergeBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), TrueBB);
  MF->insert(++TrueBB->getIterator(), MergeBB);

  // We want to copy the "true" value only when the branch is executed.
  // The SDNodeXform is responsible for the inversion.
  unsigned BranchOpCode =
      MI.getOperand(MI.getNumExplicitOperands() - 3).getImm();

  // Insert branch instruction.
  BuildMI(MBB, MBBI, DL, TII->get(BranchOpCode))
      .add(MI.getOperand(MI.getNumExplicitOperands() - 2))
      .add(MI.getOperand(MI.getNumExplicitOperands() - 1))
      .addMBB(MergeBB);

  Register DestReg = MI.getOperand(0).getReg();
  assert(MI.getOperand(1).getReg() == DestReg);

  if (MI.getOpcode() == RISCV::PseudoCCMOVGPR ||
      MI.getOpcode() == RISCV::PseudoCCMOVGPRNoX0) {
    // Add MV.
    BuildMI(TrueBB, DL, TII->get(RISCV::ADDI), DestReg)
        .add(MI.getOperand(2))
        .addImm(0);
  } else {
    unsigned NewOpc;
    // clang-format off
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case RISCV::PseudoCCADD:   NewOpc = RISCV::ADD;   break;
    case RISCV::PseudoCCSUB:   NewOpc = RISCV::SUB;   break;
    case RISCV::PseudoCCSLL:   NewOpc = RISCV::SLL;   break;
    case RISCV::PseudoCCSRL:   NewOpc = RISCV::SRL;   break;
    case RISCV::PseudoCCSRA:   NewOpc = RISCV::SRA;   break;
    case RISCV::PseudoCCAND:   NewOpc = RISCV::AND;   break;
    case RISCV::PseudoCCOR:    NewOpc = RISCV::OR;    break;
    case RISCV::PseudoCCXOR:   NewOpc = RISCV::XOR;   break;
    case RISCV::PseudoCCMAX:   NewOpc = RISCV::MAX;   break;
    case RISCV::PseudoCCMIN:   NewOpc = RISCV::MIN;   break;
    case RISCV::PseudoCCMAXU:  NewOpc = RISCV::MAXU;  break;
    case RISCV::PseudoCCMINU:  NewOpc = RISCV::MINU;  break;
    case RISCV::PseudoCCMUL:   NewOpc = RISCV::MUL;   break;
    case RISCV::PseudoCCLUI:   NewOpc = RISCV::LUI;   break;
    case RISCV::PseudoCCQC_E_LB:  NewOpc = RISCV::QC_E_LB;    break;
    case RISCV::PseudoCCQC_E_LH:  NewOpc = RISCV::QC_E_LH;    break;
    case RISCV::PseudoCCQC_E_LW:  NewOpc = RISCV::QC_E_LW;    break;
    case RISCV::PseudoCCQC_E_LHU: NewOpc = RISCV::QC_E_LHU;   break;
    case RISCV::PseudoCCQC_E_LBU: NewOpc = RISCV::QC_E_LBU;   break;
    case RISCV::PseudoCCLB:    NewOpc = RISCV::LB;    break;
    case RISCV::PseudoCCLH:    NewOpc = RISCV::LH;    break;
    case RISCV::PseudoCCLW:    NewOpc = RISCV::LW;    break;
    case RISCV::PseudoCCLHU:   NewOpc = RISCV::LHU;   break;
    case RISCV::PseudoCCLBU:   NewOpc = RISCV::LBU;   break;
    case RISCV::PseudoCCLWU:   NewOpc = RISCV::LWU;   break;
    case RISCV::PseudoCCLD:    NewOpc = RISCV::LD;    break;
    case RISCV::PseudoCCQC_LI:  NewOpc = RISCV::QC_LI;   break;
    case RISCV::PseudoCCQC_E_LI: NewOpc = RISCV::QC_E_LI;   break;
    case RISCV::PseudoCCADDI:  NewOpc = RISCV::ADDI;  break;
    case RISCV::PseudoCCSLLI:  NewOpc = RISCV::SLLI;  break;
    case RISCV::PseudoCCSRLI:  NewOpc = RISCV::SRLI;  break;
    case RISCV::PseudoCCSRAI:  NewOpc = RISCV::SRAI;  break;
    case RISCV::PseudoCCANDI:  NewOpc = RISCV::ANDI;  break;
    case RISCV::PseudoCCORI:   NewOpc = RISCV::ORI;   break;
    case RISCV::PseudoCCXORI:  NewOpc = RISCV::XORI;  break;
    case RISCV::PseudoCCADDW:  NewOpc = RISCV::ADDW;  break;
    case RISCV::PseudoCCSUBW:  NewOpc = RISCV::SUBW;  break;
    case RISCV::PseudoCCSLLW:  NewOpc = RISCV::SLLW;  break;
    case RISCV::PseudoCCSRLW:  NewOpc = RISCV::SRLW;  break;
    case RISCV::PseudoCCSRAW:  NewOpc = RISCV::SRAW;  break;
    case RISCV::PseudoCCADDIW: NewOpc = RISCV::ADDIW; break;
    case RISCV::PseudoCCSLLIW: NewOpc = RISCV::SLLIW; break;
    case RISCV::PseudoCCSRLIW: NewOpc = RISCV::SRLIW; break;
    case RISCV::PseudoCCSRAIW: NewOpc = RISCV::SRAIW; break;
    case RISCV::PseudoCCANDN:  NewOpc = RISCV::ANDN;  break;
    case RISCV::PseudoCCORN:   NewOpc = RISCV::ORN;   break;
    case RISCV::PseudoCCXNOR:  NewOpc = RISCV::XNOR;  break;
    case RISCV::PseudoCCNDS_BFOS: NewOpc = RISCV::NDS_BFOS; break;
    case RISCV::PseudoCCNDS_BFOZ: NewOpc = RISCV::NDS_BFOZ; break;
    }
    // clang-format on

    if (NewOpc == RISCV::NDS_BFOZ || NewOpc == RISCV::NDS_BFOS) {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg)
          .add(MI.getOperand(2))
          .add(MI.getOperand(3))
          .add(MI.getOperand(4));
    } else if (NewOpc == RISCV::LUI || NewOpc == RISCV::QC_LI ||
               NewOpc == RISCV::QC_E_LI) {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg).add(MI.getOperand(2));
    } else {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg)
          .add(MI.getOperand(2))
          .add(MI.getOperand(3));
    }
  }

  TrueBB->addSuccessor(MergeBB);

  MergeBB->splice(MergeBB->end(), &MBB, MI, MBB.end());
  MergeBB->transferSuccessors(&MBB);

  MBB.addSuccessor(TrueBB);
  MBB.addSuccessor(MergeBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  // Make sure live-ins are correctly attached to this new basic block.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *TrueBB);
  computeAndAddLiveIns(LiveRegs, *MergeBB);

  return true;
}

bool RISCVExpandPseudoPreEmitImpl::expandCCOpToCMov(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  if (MI.getOpcode() != RISCV::PseudoCCMOVGPR &&
      MI.getOpcode() != RISCV::PseudoCCMOVGPRNoX0)
    return false;

  if (!STI->hasVendorXqcicm())
    return false;

  MachineOperand &LHS = MI.getOperand(MI.getNumExplicitOperands() - 2);
  MachineOperand &RHS = MI.getOperand(MI.getNumExplicitOperands() - 1);

  // FIXME: Would be wonderful to support LHS=X0, but not very easy.
  if (LHS.getReg() == RISCV::X0 || MI.getOperand(1).getReg() == RISCV::X0 ||
      MI.getOperand(2).getReg() == RISCV::X0)
    return false;

  // Use branch opcode to select appropriate Xqcicm instruction
  unsigned BCC = MI.getOperand(MI.getNumExplicitOperands() - 3).getImm();
  std::optional<unsigned> CMovRegOpcode;
  bool IsSigned = true;
  unsigned CMovImmOpcode;
  switch (BCC) {
  default:
    return false; // Unhandled branch opcodes
  case RISCV::BNE:
    CMovRegOpcode = RISCV::QC_MVEQ;
    CMovImmOpcode = RISCV::QC_MVEQI;
    break;
  case RISCV::BEQ:
    CMovRegOpcode = RISCV::QC_MVNE;
    CMovImmOpcode = RISCV::QC_MVNEI;
    break;
  case RISCV::BGE:
    CMovRegOpcode = RISCV::QC_MVLT;
    CMovImmOpcode = RISCV::QC_MVLTI;
    break;
  case RISCV::BLT:
    CMovRegOpcode = RISCV::QC_MVGE;
    CMovImmOpcode = RISCV::QC_MVGEI;
    break;
  case RISCV::BGEU:
    CMovRegOpcode = RISCV::QC_MVLTU;
    CMovImmOpcode = RISCV::QC_MVLTUI;
    break;
  case RISCV::BLTU:
    CMovRegOpcode = RISCV::QC_MVGEU;
    CMovImmOpcode = RISCV::QC_MVGEUI;
    break;
  case RISCV::QC_BEQI:
    CMovImmOpcode = RISCV::QC_MVNEI;
    break;
  case RISCV::QC_BNEI:
    CMovImmOpcode = RISCV::QC_MVEQI;
    break;
  case RISCV::QC_BLTI:
    CMovImmOpcode = RISCV::QC_MVGEI;
    break;
  case RISCV::QC_BGEI:
    CMovImmOpcode = RISCV::QC_MVLTI;
    break;
  case RISCV::QC_BLTUI:
    CMovImmOpcode = RISCV::QC_MVGEUI;
    IsSigned = false;
    break;
  case RISCV::QC_BGEUI:
    CMovImmOpcode = RISCV::QC_MVLTUI;
    IsSigned = false;
    break;
  }

  if (RHS.isImm()) {
    if ((!isInt<5>(RHS.getImm()) || !IsSigned) &&
        (!isUInt<5>(RHS.getImm()) || IsSigned))
      return false;

    // $dst = PseudoCCMOVGPR $falsev(=$dst), $truev, $opcode, $lhs, $rhs_imm
    // $dst = PseudoCCMOVGPRNoX0 $falsev(=$dst), $truev, $opcode, $lhs, $rhs_imm
    // =>
    // $dst = QC_MVccI $falsev (=$dst), $lhs, $rhs_imm, $truev
    BuildMI(MBB, MBBI, DL, TII->get(CMovImmOpcode))
        .addDef(MI.getOperand(0).getReg())
        .addReg(MI.getOperand(1).getReg())
        .addReg(LHS.getReg())
        .add(RHS)
        .addReg(MI.getOperand(2).getReg());

    MI.eraseFromParent();
    return true;
  }

  if (RHS.getReg() == RISCV::X0) {
    // $dst = PseudoCCMOVGPR $falsev (=$dst), $truev, $opcode, $lhs, X0
    // $dst = PseudoCCMOVGPRNoX0 $falsev (=$dst), $truev, $opcode, $lhs, X0
    // =>
    // $dst = QC_MVccI $falsev (=$dst), $lhs, 0, $truev
    BuildMI(MBB, MBBI, DL, TII->get(CMovImmOpcode))
        .addDef(MI.getOperand(0).getReg())
        .addReg(MI.getOperand(1).getReg())
        .addReg(LHS.getReg())
        .addImm(0)
        .addReg(MI.getOperand(2).getReg());

    MI.eraseFromParent();
    return true;
  }

  if (!CMovRegOpcode)
    return false;

  // $dst = PseudoCCMOVGPR $falsev (=$dst), $truev, $opcode, $lhs, $rhs
  // $dst = PseudoCCMOVGPRNoX0 $falsev (=$dst), $truev, $opcode, $lhs, $rhs
  // =>
  // $dst = QC_MVcc $falsev (=$dst), $lhs, $rhs, $truev
  BuildMI(MBB, MBBI, DL, TII->get(*CMovRegOpcode))
      .addDef(MI.getOperand(0).getReg())
      .addReg(MI.getOperand(1).getReg())
      .addReg(LHS.getReg())
      .addReg(RHS.getReg())
      .addReg(MI.getOperand(2).getReg());
  MI.eraseFromParent();
  return true;
}

bool RISCVExpandPseudoPreEmitImpl::expandVMSET_VMCLR(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    unsigned Opcode) const {
  DebugLoc DL = MBBI->getDebugLoc();
  Register DstReg = MBBI->getOperand(0).getReg();
  const MCInstrDesc &Desc = TII->get(Opcode);
  BuildMI(MBB, MBBI, DL, Desc, DstReg)
      .addReg(DstReg, RegState::Undef)
      .addReg(DstReg, RegState::Undef);
  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool RISCVExpandPseudoPreEmitImpl::expandMV_FPR16INX(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register DstReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(0).getReg(), RISCV::sub_16, &RISCV::GPRRegClass);
  Register SrcReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(1).getReg(), RISCV::sub_16, &RISCV::GPRRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(MBBI->getOperand(1).isKill()))
      .addImm(0);

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool RISCVExpandPseudoPreEmitImpl::expandMV_FPR32INX(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register DstReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(0).getReg(), RISCV::sub_32, &RISCV::GPRRegClass);
  Register SrcReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(1).getReg(), RISCV::sub_32, &RISCV::GPRRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(MBBI->getOperand(1).isKill()))
      .addImm(0);

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

// This function expands the PseudoRV32ZdinxSD for storing a double-precision
// floating-point value into memory by generating an equivalent instruction
// sequence for RV32.
bool RISCVExpandPseudoPreEmitImpl::expandRV32ZdinxStore(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register Lo =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_even);
  Register Hi =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_odd);
  if (Hi == RISCV::DUMMY_REG_PAIR_WITH_X0)
    Hi = RISCV::X0;

  auto MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                   .addReg(Lo, getKillRegState(MBBI->getOperand(0).isKill()))
                   .addReg(MBBI->getOperand(1).getReg())
                   .add(MBBI->getOperand(2));

  MachineInstrBuilder MIBHi;
  if (MBBI->getOperand(2).isGlobal() || MBBI->getOperand(2).isCPI()) {
    assert(MBBI->getOperand(2).getOffset() % 8 == 0);
    MBBI->getOperand(2).setOffset(MBBI->getOperand(2).getOffset() + 4);
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                .addReg(Hi, getKillRegState(MBBI->getOperand(0).isKill()))
                .add(MBBI->getOperand(1))
                .add(MBBI->getOperand(2));
  } else {
    assert(isInt<12>(MBBI->getOperand(2).getImm() + 4));
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                .addReg(Hi, getKillRegState(MBBI->getOperand(0).isKill()))
                .add(MBBI->getOperand(1))
                .addImm(MBBI->getOperand(2).getImm() + 4);
  }

  MachineFunction *MF = MBB.getParent();
  SmallVector<MachineMemOperand *> NewLoMMOs;
  SmallVector<MachineMemOperand *> NewHiMMOs;
  for (const MachineMemOperand *MMO : MBBI->memoperands()) {
    NewLoMMOs.push_back(MF->getMachineMemOperand(MMO, 0, 4));
    NewHiMMOs.push_back(MF->getMachineMemOperand(MMO, 4, 4));
  }
  MIBLo.setMemRefs(NewLoMMOs);
  MIBHi.setMemRefs(NewHiMMOs);

  MBBI->eraseFromParent();
  return true;
}

// This function expands PseudoRV32ZdinxLoad for loading a double-precision
// floating-point value from memory into an equivalent instruction sequence for
// RV32.
bool RISCVExpandPseudoPreEmitImpl::expandRV32ZdinxLoad(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register Lo =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_even);
  Register Hi =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_odd);
  assert(Hi != RISCV::DUMMY_REG_PAIR_WITH_X0 && "Cannot write to X0_Pair");

  MachineInstrBuilder MIBLo, MIBHi;

  // If the register of operand 1 is equal to the Lo register, then swap the
  // order of loading the Lo and Hi statements.
  bool IsOp1EqualToLo = Lo == MBBI->getOperand(1).getReg();
  // Order: Lo, Hi
  if (!IsOp1EqualToLo) {
    MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Lo)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
  }

  if (MBBI->getOperand(2).isGlobal() || MBBI->getOperand(2).isCPI()) {
    auto Offset = MBBI->getOperand(2).getOffset();
    assert(Offset % 8 == 0);
    MBBI->getOperand(2).setOffset(Offset + 4);
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Hi)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
    MBBI->getOperand(2).setOffset(Offset);
  } else {
    assert(isInt<12>(MBBI->getOperand(2).getImm() + 4));
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Hi)
                .addReg(MBBI->getOperand(1).getReg())
                .addImm(MBBI->getOperand(2).getImm() + 4);
  }

  // Order: Hi, Lo
  if (IsOp1EqualToLo) {
    MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Lo)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
  }

  MachineFunction *MF = MBB.getParent();
  SmallVector<MachineMemOperand *> NewLoMMOs;
  SmallVector<MachineMemOperand *> NewHiMMOs;
  for (const MachineMemOperand *MMO : MBBI->memoperands()) {
    NewLoMMOs.push_back(MF->getMachineMemOperand(MMO, 0, 4));
    NewHiMMOs.push_back(MF->getMachineMemOperand(MMO, 4, 4));
  }
  MIBLo.setMemRefs(NewLoMMOs);
  MIBHi.setMemRefs(NewHiMMOs);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudoPreEmitImpl::expandPseudoReadVLENBViaVSETVLIX0(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  DebugLoc DL = MBBI->getDebugLoc();
  Register Dst = MBBI->getOperand(0).getReg();
  unsigned Mul = MBBI->getOperand(1).getImm();
  RISCVVType::VLMUL VLMUL = RISCVVType::encodeLMUL(Mul, /*Fractional=*/false);
  unsigned VTypeImm = RISCVVType::encodeVTYPE(
      VLMUL, /*SEW=*/8, /*TailAgnostic=*/true, /*MaskAgnostic=*/true);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoVSETVLIX0))
      .addReg(Dst, RegState::Define)
      .addReg(RISCV::X0, RegState::Kill)
      .addImm(VTypeImm);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudoPreEmitImpl::expandPseudoClearFPR64(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) const {
  const DebugLoc &DL = MBBI->getDebugLoc();
  Register Dst = MBBI->getOperand(0).getReg();

  if (STI->is64Bit()) {
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::FMV_D_X), Dst).addReg(RISCV::X0);
  } else {
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::FCVT_D_W), Dst)
        .addReg(RISCV::X0)
        .addImm(RISCVFPRndMode::RNE);
  }

  MBBI->eraseFromParent();
  return true;
}

char RISCVExpandPseudoPreEmitLegacy::ID = 0;

INITIALIZE_PASS(RISCVExpandPseudoPreEmitLegacy, "riscv-expand-pseudo-pre-emit",
                RISCV_EXPAND_PSEUDO_PRE_EMIT_NAME, false, false)

FunctionPass *llvm::createRISCVExpandPseudoPreEmitLegacyPass() {
  return new RISCVExpandPseudoPreEmitLegacy();
}

PreservedAnalyses
RISCVExpandPseudoPreEmitPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  bool Changed = RISCVExpandPseudoPreEmitImpl().run(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses();
}
