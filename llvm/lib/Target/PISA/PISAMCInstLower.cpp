//===-- PISAMCInstLower.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAMCInstLower.h"
#include "MCTargetDesc/PISAMCExpr.h"
#include "PISA.h"
#include "PISARegManager.h"
#include "PISASubtarget.h"
#include "PISAUtils.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

MCOperand PISAMCInstLower::lowerSymbolOperand(const MachineOperand &MO,
                                              MCSymbol &Sym) const {
  const MCExpr *Expr = PISAGlobalAddressMCExpr::create(Sym, OutContext);
  return MCOperand::createExpr(Expr);
}

MCSymbol &
PISAMCInstLower::getGlobalAddressSymbol(const MachineOperand &MO) const {
  assert(MO.isGlobal());
  return *AP.getSymbol(MO.getGlobal());
}

void PISAMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());
  for (unsigned OpNo = 0, E = MI->getNumOperands(); OpNo != E; ++OpNo) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Metadata: {
      // Skip call to addOperand() outside the switch as MCInst doesn't
      // support metadata operand types.
      continue;
    }
    case MachineOperand::MO_ExternalSymbol: {
      // Convert external symbol operand (kernel arg name on loadParam
      // instructions) to an MCExpr for PISAInstPrinter::printParamMemOperand.
      MCSymbol *Sym = OutContext.getOrCreateSymbol(MO.getSymbolName());
      MCOp = MCOperand::createExpr(MCSymbolRefExpr::create(Sym, OutContext));
      break;
    }
    case MachineOperand::MO_FrameIndex: {
      // MachineOperand of type MO_FrameIndex is lowered to immediate operand
      // with value equal to frame index. Whether an immediate operand is
      // frame index is indicated by flags that are stored in MCInst instance.
      // During operand printing, we inspect flag value first and decide if
      // the operand has to be printed as private variable or a generic
      // immediate value.
      unsigned int FrameIndex = MO.getIndex();
      MCOp = MCOperand::createImm(FrameIndex);
      setVariableRef(OutMI, OpNo);
      break;
    }
    case MachineOperand::MO_GlobalAddress: {
      MCOp = lowerSymbolOperand(MO, getGlobalAddressSymbol(MO));
      break;
    }
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::createExpr(
          MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), OutContext));
      break;
    case MachineOperand::MO_Register: {
      Register CurReg = MO.getReg();
      unsigned EncodedVal = CurReg;
      if (CurReg.isVirtual()) {
        auto &MRI = MI->getParent()->getParent()->getRegInfo();
        auto *RC = MRI.getRegClass(CurReg);
        auto NumElts = TRI.getNumEltsFromRegClass(RC);
        auto EltSize = TRI.getBitSizeFromRegClass(RC);
        auto Bank = RegMgr.getRegBank(NumElts, EltSize);
        EncodedVal = RegMgr.encodeVirtualRegister(Bank, CurReg);
      }
      MCOp = MCOperand::createReg(EncodedVal);
      auto SubReg = MO.getSubReg();
      auto IsMov = MI->isMoveImmediate() || MI->isMoveReg();
      auto Swizzle = TRI.getSwizzle(SubReg);
      if (IsMov && !SubReg && CurReg.isVirtual()) {
        // vector args in mov instructions always print swizzle
        auto &MRI = MI->getParent()->getParent()->getRegInfo();
        auto *RC = MRI.getRegClass(CurReg);
        auto NumElts = TRI.getNumEltsFromRegClass(RC);
        if (NumElts == 2)
          Swizzle = PISA::Swizzle::XY;
        else if (NumElts == 4)
          Swizzle = PISA::Swizzle::XYZW;
        else
          assert((NumElts == 1) && "unknown swizzle value");
      }
      setSwizzle(OutMI, OpNo, Swizzle);
      break;
    }
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    case MachineOperand::MO_FPImmediate:
      // All floating point values are bitcasted
      // into integer ones. Double- and single-precison ones could be specially
      // taged in MC as SFP or DFP immediate. But, in general, as each type of
      // immediate operand has its own methods, general immediate operands
      // won't loss any semantics.
      uint64_t ImmVal =
          MO.getFPImm()->getValueAPF().bitcastToAPInt().getZExtValue();
      // TODO: bfloat16 has a different exponent/mantissa layout than
      // IEEE half; a separate kBFPImmediate kind may be needed to
      // distinguish them (see also getFPImmInReg TODO).
      if (MO.getFPImm()->getType()->isHalfTy() ||
          MO.getFPImm()->getType()->isBFloatTy())
        MCOp = MCOperand::createHFPImm(ImmVal);
      else if (MO.getFPImm()->getType()->isFloatTy())
        MCOp = MCOperand::createSFPImm(ImmVal);
      else if (MO.getFPImm()->getType()->isDoubleTy())
        MCOp = MCOperand::createDFPImm(ImmVal);
      else
        MCOp = MCOperand::createImm(ImmVal);
      break;
    }

    OutMI.addOperand(MCOp);
  }
}
