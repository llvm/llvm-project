//===-- RISCVMCInstLower.cpp - Convert RISCV MachineInstr to an MCInst ------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower RISCV MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "MCTargetDesc/RISCVMCExpr.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static MCOperand lowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym,
                                    const AsmPrinter &AP) {
  MCContext &Ctx = AP.OutContext;
  RISCVMCExpr::VariantKind Kind;

  switch (MO.getTargetFlags()) {
  default:
    llvm_unreachable("Unknown target flag on GV operand");
  case RISCVII::MO_None:
    Kind = RISCVMCExpr::VK_RISCV_None;
    break;
  case RISCVII::MO_CALL:
    Kind = RISCVMCExpr::VK_RISCV_CALL;
    break;
  case RISCVII::MO_PLT:
    Kind = RISCVMCExpr::VK_RISCV_CALL_PLT;
    break;
  case RISCVII::MO_LO:
    Kind = RISCVMCExpr::VK_RISCV_LO;
    break;
  case RISCVII::MO_HI:
    Kind = RISCVMCExpr::VK_RISCV_HI;
    break;
  case RISCVII::MO_PCREL_LO:
    Kind = RISCVMCExpr::VK_RISCV_PCREL_LO;
    break;
  case RISCVII::MO_PCREL_HI:
    Kind = RISCVMCExpr::VK_RISCV_PCREL_HI;
    break;
  case RISCVII::MO_GOT_HI:
    Kind = RISCVMCExpr::VK_RISCV_GOT_HI;
    break;
  case RISCVII::MO_TPREL_LO:
    Kind = RISCVMCExpr::VK_RISCV_TPREL_LO;
    break;
  case RISCVII::MO_TPREL_HI:
    Kind = RISCVMCExpr::VK_RISCV_TPREL_HI;
    break;
  case RISCVII::MO_TPREL_ADD:
    Kind = RISCVMCExpr::VK_RISCV_TPREL_ADD;
    break;
  case RISCVII::MO_TLS_GOT_HI:
    Kind = RISCVMCExpr::VK_RISCV_TLS_GOT_HI;
    break;
  case RISCVII::MO_TLS_GD_HI:
    Kind = RISCVMCExpr::VK_RISCV_TLS_GD_HI;
    break;
  }

  const MCExpr *ME =
      MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    ME = MCBinaryExpr::createAdd(
        ME, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);

  if (Kind != RISCVMCExpr::VK_RISCV_None)
    ME = RISCVMCExpr::create(ME, Kind, Ctx);
  return MCOperand::createExpr(ME);
}

bool llvm::lowerRISCVMachineOperandToMCOperand(const MachineOperand &MO,
                                               MCOperand &MCOp,
                                               const AsmPrinter &AP) {
  switch (MO.getType()) {
  default:
    report_fatal_error("LowerRISCVMachineInstrToMCInst: unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return false;
    MCOp = MCOperand::createReg(MO.getReg());
    break;
  case MachineOperand::MO_RegisterMask:
    // Regmasks are like implicit defs.
    return false;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::createImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = lowerSymbolOperand(MO, MO.getMBB()->getSymbol(), AP);
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = lowerSymbolOperand(MO, AP.getSymbolPreferLocal(*MO.getGlobal()), AP);
    break;
  case MachineOperand::MO_BlockAddress:
    MCOp = lowerSymbolOperand(
        MO, AP.GetBlockAddressSymbol(MO.getBlockAddress()), AP);
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = lowerSymbolOperand(
        MO, AP.GetExternalSymbolSymbol(MO.getSymbolName()), AP);
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = lowerSymbolOperand(MO, AP.GetCPISymbol(MO.getIndex()), AP);
    break;
  case MachineOperand::MO_JumpTableIndex:
    MCOp = lowerSymbolOperand(MO, AP.GetJTISymbol(MO.getIndex()), AP);
    break;
  }
  return true;
}

static bool lowerRISCVVMachineInstrToMCInst(const MachineInstr *MI,
                                            MCInst &OutMI) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI->getOpcode());
  if (!RVV)
    return false;

  OutMI.setOpcode(RVV->BaseInstr);

  const MachineBasicBlock *MBB = MI->getParent();
  assert(MBB && "MI expected to be in a basic block");
  const MachineFunction *MF = MBB->getParent();
  assert(MF && "MBB expected to be in a machine function");

  const TargetRegisterInfo *TRI =
      MF->getSubtarget<RISCVSubtarget>().getRegisterInfo();
  assert(TRI && "TargetRegisterInfo expected");

  uint64_t TSFlags = MI->getDesc().TSFlags;
  unsigned NumOps = MI->getNumExplicitOperands();

  // Skip policy, VL and SEW operands which are the last operands if present.
  if (RISCVII::hasVecPolicyOp(TSFlags))
    --NumOps;
  if (RISCVII::hasVLOp(TSFlags))
    --NumOps;
  if (RISCVII::hasSEWOp(TSFlags))
    --NumOps;

  for (unsigned OpNo = 0; OpNo != NumOps; ++OpNo) {
    const MachineOperand &MO = MI->getOperand(OpNo);

    // Skip merge op. It should be the first operand after the result.
    if (RISCVII::hasMergeOp(TSFlags) && OpNo == 1) {
      assert(MI->getNumExplicitDefs() == 1);
      continue;
    }

    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      llvm_unreachable("Unknown operand type");
    case MachineOperand::MO_Register: {
      Register Reg = MO.getReg();

      if (RISCV::VRM2RegClass.contains(Reg) ||
          RISCV::VRM4RegClass.contains(Reg) ||
          RISCV::VRM8RegClass.contains(Reg)) {
        Reg = TRI->getSubReg(Reg, RISCV::sub_vrm1_0);
        assert(Reg && "Subregister does not exist");
      } else if (RISCV::FPR16RegClass.contains(Reg)) {
        Reg = TRI->getMatchingSuperReg(Reg, RISCV::sub_16, &RISCV::FPR32RegClass);
        assert(Reg && "Subregister does not exist");
      } else if (RISCV::FPR64RegClass.contains(Reg)) {
        Reg = TRI->getSubReg(Reg, RISCV::sub_32);
        assert(Reg && "Superregister does not exist");
      }

      MCOp = MCOperand::createReg(Reg);
      break;
    }
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    }
    OutMI.addOperand(MCOp);
  }

  // Unmasked pseudo instructions need to append dummy mask operand to
  // V instructions. All V instructions are modeled as the masked version.
  if (RISCVII::hasDummyMaskOp(TSFlags))
    OutMI.addOperand(MCOperand::createReg(RISCV::NoRegister));

  return true;
}

bool llvm::lowerRISCVMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                          AsmPrinter &AP) {
  if (lowerRISCVVMachineInstrToMCInst(MI, OutMI))
    return false;

  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp;
    if (lowerRISCVMachineOperandToMCOperand(MO, MCOp, AP))
      OutMI.addOperand(MCOp);
  }

  switch (OutMI.getOpcode()) {
  case TargetOpcode::PATCHABLE_FUNCTION_ENTER: {
    const Function &F = MI->getParent()->getParent()->getFunction();
    if (F.hasFnAttribute("patchable-function-entry")) {
      unsigned Num;
      if (F.getFnAttribute("patchable-function-entry")
              .getValueAsString()
              .getAsInteger(10, Num))
        return false;
      AP.emitNops(Num);
      return true;
    }
    break;
  }
  case RISCV::PseudoReadVLENB:
    OutMI.setOpcode(RISCV::CSRRS);
    OutMI.addOperand(MCOperand::createImm(
        RISCVSysReg::lookupSysRegByName("VLENB")->Encoding));
    OutMI.addOperand(MCOperand::createReg(RISCV::X0));
    break;
  case RISCV::PseudoReadVL:
    OutMI.setOpcode(RISCV::CSRRS);
    OutMI.addOperand(
        MCOperand::createImm(RISCVSysReg::lookupSysRegByName("VL")->Encoding));
    OutMI.addOperand(MCOperand::createReg(RISCV::X0));
    break;
  }
  return false;
}
