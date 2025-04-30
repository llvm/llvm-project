//===-- Next32MCInstLower.h - Lower MachineInstr to MCInst ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Next32_Next32MCINSTLOWER_H
#define LLVM_LIB_TARGET_Next32_Next32MCINSTLOWER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
class AsmPrinter;
class MCContext;
class MCInst;
class MCOperand;
class MCSymbol;
class MachineInstr;
class MachineModuleInfoMachO;
class MachineOperand;
class Mangler;

// Next32MCInstLower - This class is used to lower an MachineInstr into an
// MCInst.
class LLVM_LIBRARY_VISIBILITY Next32MCInstLower {
  MCContext &Ctx;

  AsmPrinter &Printer;

public:
  Next32MCInstLower(MCContext &ctx, AsmPrinter &printer)
      : Ctx(ctx), Printer(printer) {}
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;
  MCOperand LowerGlobalAddress(const MachineOperand &MO) const;
};
} // namespace llvm

#endif
