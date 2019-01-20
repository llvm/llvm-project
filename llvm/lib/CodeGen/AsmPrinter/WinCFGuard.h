//===-- WinCFGuard.h - Windows Control Flow Guard Handling ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing windows exception info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_WINCFGUARD_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_WINCFGUARD_H

#include "llvm/CodeGen/AsmPrinterHandler.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class LLVM_LIBRARY_VISIBILITY WinCFGuard : public AsmPrinterHandler {
  /// Target of directive emission.
  AsmPrinter *Asm;

public:
  WinCFGuard(AsmPrinter *A);
  ~WinCFGuard() override;

  void setSymbolSize(const MCSymbol *Sym, uint64_t Size) override {}

  /// Emit the Control Flow Guard function ID table
  void endModule() override;

  /// Gather pre-function debug information.
  /// Every beginFunction(MF) call should be followed by an endFunction(MF)
  /// call.
  void beginFunction(const MachineFunction *MF) override {}

  /// Gather post-function debug information.
  /// Please note that some AsmPrinter implementations may not call
  /// beginFunction at all.
  void endFunction(const MachineFunction *MF) override {}

  /// Process beginning of an instruction.
  void beginInstruction(const MachineInstr *MI) override {}

  /// Process end of an instruction.
  void endInstruction() override {}
};

} // namespace llvm

#endif
