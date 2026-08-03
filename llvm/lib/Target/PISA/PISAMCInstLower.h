//===-- PISAMCInstLower.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAMCINSTLOWER_H
#define LLVM_LIB_TARGET_PISA_PISAMCINSTLOWER_H

#include "PISADefines.h"
#include "PISARegisterInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Compiler.h"
#include <cassert>

namespace llvm {
class MCContext;
class MCOperand;
class MCSymbol;
class MachineInstr;
class MachineOperand;

namespace PISA {
class RegManager;
} // namespace PISA

struct PISAMCOpnd {
  // given immediate value represents the index of a variable (.e.g, @R0).
  bool IsVariable;
  // swizzle of given operand
  unsigned Swizzle;
};

class PISAMCInst : public MCInst {
public:
  SmallDenseMap<unsigned, PISAMCOpnd> Opnds;

  // Source line number in the original assembly file, used for debug info and
  // error reporting
  unsigned SourceLine = 0;
  unsigned getSourceLine() const { return SourceLine; }
  void setSourceLine(unsigned L) { SourceLine = L; }
};

// This class is used to lower a MachineInstr into an MCInst.
class LLVM_LIBRARY_VISIBILITY PISAMCInstLower {
public:
  static void setVariableRef(MCInst &MI, unsigned OpNo) {
    auto *MC = static_cast<PISAMCInst *>(&MI);
    auto It = MC->Opnds.find(OpNo);
    if (It == MC->Opnds.end())
      MC->Opnds[OpNo] = {true, 0};
    else
      It->second.IsVariable = true;
  }

  static bool isVariableRef(const MCInst &MI, unsigned int OpNo) {
    const auto *MC = static_cast<const PISAMCInst *>(&MI);
    auto It = MC->Opnds.find(OpNo);
    if (It == MC->Opnds.end())
      return false;
    return It->second.IsVariable;
  }

  static void setSwizzle(MCInst &MI, unsigned OpNo, PISA::Swizzle Swizzle) {
    static_assert(static_cast<unsigned>(PISA::Swizzle::NONE) <= 7);
    auto *MC = static_cast<PISAMCInst *>(&MI);
    auto It = MC->Opnds.find(OpNo);
    if (It == MC->Opnds.end())
      MC->Opnds[OpNo] = {false, static_cast<unsigned>(Swizzle)};
    else
      It->second.Swizzle = static_cast<unsigned>(Swizzle);
  }

  static PISA::Swizzle getSwizzle(const MCInst &MI, unsigned OpNo) {
    const auto *MC = static_cast<const PISAMCInst *>(&MI);
    auto It = MC->Opnds.find(OpNo);
    if (It == MC->Opnds.end())
      return static_cast<PISA::Swizzle>(PISA::Swizzle::NONE);
    return static_cast<PISA::Swizzle>(It->second.Swizzle);
  }

  PISAMCInstLower(MCContext &Ctx, const PISARegisterInfo &TRI,
                  const PISA::RegManager &RegMgr, const AsmPrinter &AP)
      : OutContext(Ctx), TRI(TRI), RegMgr(RegMgr), AP(AP) {}
  void lower(const MachineInstr *MI, MCInst &OutMI) const;

private:
  MCOperand lowerSymbolOperand(const MachineOperand &MO, MCSymbol &Sym) const;
  MCSymbol &getGlobalAddressSymbol(const MachineOperand &MO) const;

private:
  MCContext &OutContext;
  const PISARegisterInfo &TRI;
  const PISA::RegManager &RegMgr;
  const AsmPrinter &AP;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAMCINSTLOWER_H
