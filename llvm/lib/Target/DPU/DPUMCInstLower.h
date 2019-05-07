//===-- DPUMCInstLower.h - Lower MachineInstr to MCInst ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUMCINSTLOWER_H
#define LLVM_LIB_TARGET_DPU_DPUMCINSTLOWER_H

#include "llvm/Support/Compiler.h"
#include <llvm/CodeGen/MachineOperand.h>

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

// DPUMCInstLower - This class is used to lower an MachineInstr into an MCInst.
class LLVM_LIBRARY_VISIBILITY DPUMCInstLower {
  MCContext &Ctx;

  AsmPrinter &Printer;

public:
  DPUMCInstLower(MCContext &ctx, AsmPrinter &printer)
      : Ctx(ctx), Printer(printer) {}
  void Lower(const MachineInstr *MI, MCInst &OutMI) const;

  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;

  MCSymbol *GetGlobalAddressSymbol(const MachineOperand &MO) const;
  MCSymbol *GetExternalSymbol(const MachineOperand &MO) const;
  MCSymbol *GetJumpTableSymbol(const MachineOperand &MO) const;
};
} // namespace llvm

#endif
