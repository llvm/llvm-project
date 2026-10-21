#include "GenpayMCInstLower.h"
#include "GenpayMCTargetDesc.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

void GenpayMCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  if (MI->getOpcode() == TargetOpcode::COPY) {
    OutMI.setOpcode(Genpay::MOVrr);
    for (const MachineOperand &MO : MI->explicit_operands())
      OutMI.addOperand(LowerOperand(MO));
    return;
  }

  // Only the explicit operands map to the MC layer; implicit chain/glue and
  // other non-MC operands are dropped here.
  OutMI.setOpcode(MI->getOpcode());
  for (const MachineOperand &MO : MI->explicit_operands()) {
    MCOperand MCOp = LowerOperand(MO);
    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}

MCOperand GenpayMCInstLower::LowerOperand(const MachineOperand &MO) const {
  switch (MO.getType()) {
  default:
    llvm_unreachable("Genpay: unknown operand type");
  case MachineOperand::MO_Register:
    return MCOperand::createReg(MO.getReg());
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm());
  }
}
