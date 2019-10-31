//===- XtensaMCInstLower.h - Lower MachineInstr to MCInst ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSAMCINSTLOWER_H
#define LLVM_LIB_TARGET_XTENSA_XTENSAMCINSTLOWER_H

#include "XtensaAsmPrinter.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCContext;
class MCInst;
class MCOperand;
class MCSymbol;
class MachineInstr;
class MachineOperand;
class XtensaAsmPrinter;

class LLVM_LIBRARY_VISIBILITY XtensaMCInstLower {
  MCContext &Ctx;
  XtensaAsmPrinter &Printer;

public:
  XtensaMCInstLower(MCContext &ctx, XtensaAsmPrinter &asmPrinter);

  // Lower MachineInstr MI to MCInst OutMI.
  void lower(const MachineInstr *MI, MCInst &OutMI) const;

  // Return an MCOperand for MO.  Return an empty operand if MO is implicit.
  MCOperand lowerOperand(const MachineOperand &MO, unsigned Offset = 0) const;

private:
  MCSymbol *GetConstantPoolIndexSymbol(const MachineOperand &MO) const;

  MCOperand LowerSymbolOperand(const MachineOperand &MO,
                               MachineOperand::MachineOperandType MOTy,
                               unsigned Offset) const;
};
} // end namespace llvm

#endif /* LLVM_LIB_TARGET_XTENSA_XTENSAMCINSTLOWER_H */
