//===-- AGXMCInstLower.cpp - Lower MachineInstr to MCInst -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pseudo instructions have the operands of their real instruction, except
// that real ALU destinations carry a cache hint immediate. The hint is
// inserted here (as zero: no hint).
//
//===----------------------------------------------------------------------===//

#include "AGXMCInstLower.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

bool AGXMCInstLower::lowerOperand(const MachineOperand &MO,
                                  MCOperand &MCOp) const {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    if (MO.isImplicit())
      return false;
    MCOp = MCOperand::createReg(MO.getReg());
    return true;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::createImm(MO.getImm());
    return true;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = MCOperand::createExpr(
        MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), Ctx));
    return true;
  case MachineOperand::MO_RegisterMask:
    return false;
  default:
    report_fatal_error("unsupported AGX machine operand");
  }
}

bool AGXMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  const AGXInstrInfo *TII = ST.getInstrInfo();
  int Opcode = TII->pseudoToMCOpcode(MI->getOpcode());
  if (Opcode < 0)
    return false;
  OutMI.setOpcode(Opcode);

  // mov_imm r0l, n.
  if (MI->getOpcode() == AGX::SET_NEST) {
    OutMI.addOperand(MCOperand::createReg(AGX::R0L));
    OutMI.addOperand(MCOperand::createImm(0));
    OutMI.addOperand(MCOperand::createImm(MI->getOperand(0).getImm()));
    return true;
  }

  const MCInstrDesc &PseudoDesc = MI->getDesc();
  const MCInstrDesc &RealDesc = TII->get(Opcode);
  bool InsertCache = PseudoDesc.getNumDefs() > 0 &&
                     RealDesc.getNumOperands() == PseudoDesc.getNumOperands() + 1;

  // Atomics whose result is unused write no register.
  if (MI->getOpcode() == AGX::DEVICE_ATOMIC64_NORET ||
      MI->getOpcode() == AGX::DEVICE_ATOMIC_NORET)
    OutMI.addOperand(MCOperand::createReg(MCRegister()));

  unsigned Idx = 0;
  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp;
    if (!lowerOperand(MO, MCOp))
      continue;
    OutMI.addOperand(MCOp);
    if (Idx++ == 0 && InsertCache)
      OutMI.addOperand(MCOperand::createImm(0));
  }
  // Trailing flag operands of the real instruction that code generation does
  // not set.
  if (Opcode == AGX::DEVICE_STORE_g13)
    OutMI.addOperand(MCOperand::createImm(0));
  // Device atomics on parts with several dies must bypass the per-die
  // caches.
  if ((MI->getOpcode() == AGX::DEVICE_ATOMIC ||
       MI->getOpcode() == AGX::DEVICE_ATOMIC_CMPXCHG ||
       MI->getOpcode() == AGX::DEVICE_ATOMIC64_NORET ||
       MI->getOpcode() == AGX::DEVICE_ATOMIC_NORET) &&
      ST.hasMultiDieCoherency()) {
    int Idx = AGX::getNamedOperandIdx(Opcode, AGX::OpName::coherent);
    OutMI.getOperand(Idx).setImm(OutMI.getOperand(Idx).getImm() | 1);
  }

  assert(OutMI.getNumOperands() == RealDesc.getNumOperands() &&
         "pseudo and real instruction operands do not match");
  return true;
}
