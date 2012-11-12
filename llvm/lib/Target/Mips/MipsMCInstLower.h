//===-- MipsMCInstLower.h - Lower MachineInstr to MCInst -------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSMCINSTLOWER_H
#define MIPSMCINSTLOWER_H
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
  class MCContext;
  class MCInst;
  class MCOperand;
  class MachineInstr;
  class MachineFunction;
  class Mangler;
  class MipsAsmPrinter;

/// MipsMCInstLower - This class is used to lower an MachineInstr into an
//                    MCInst.
class LLVM_LIBRARY_VISIBILITY MipsMCInstLower {
  typedef MachineOperand::MachineOperandType MachineOperandType;
  MCContext *Ctx;
  Mangler *Mang;
  MipsAsmPrinter &AsmPrinter;
public:
  MipsMCInstLower(MipsAsmPrinter &asmprinter);
  void Initialize(Mangler *mang, MCContext *C);
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;
  MCOperand LowerOperand(const MachineOperand& MO, unsigned offset = 0) const;

private:
  MCOperand LowerSymbolOperand(const MachineOperand &MO,
                               MachineOperandType MOTy, unsigned Offset) const;
};
}

#endif
