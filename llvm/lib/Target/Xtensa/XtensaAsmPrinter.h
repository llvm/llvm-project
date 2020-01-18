//===- XtensaAsmPrinter.h - Xtensa LLVM Assembly Printer --------*- C++-*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Xtensa Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSAASMPRINTER_H
#define LLVM_LIB_TARGET_XTENSA_XTENSAASMPRINTER_H

#include "XtensaTargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCStreamer;
class MachineBasicBlock;
class MachineInstr;
class Module;
class raw_ostream;

class LLVM_LIBRARY_VISIBILITY XtensaAsmPrinter : public AsmPrinter {
private:
public:
  XtensaAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  // Override AsmPrinter.
  StringRef getPassName() const override { return "Xtensa Assembly Printer"; }
  void EmitInstruction(const MachineInstr *MI) override;
  void EmitConstantPool() override;
  void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;
  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &O);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &OS) override;
  void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);
};
} // end namespace llvm

#endif /* LLVM_LIB_TARGET_XTENSA_XTENSAASMPRINTER_H */
