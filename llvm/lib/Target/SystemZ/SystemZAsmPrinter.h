//===-- SystemZAsmPrinter.h - SystemZ LLVM assembly printer ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZASMPRINTER_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZASMPRINTER_H

#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCStreamer;
class MachineBasicBlock;
class MachineInstr;
class Module;
class raw_ostream;

class LLVM_LIBRARY_VISIBILITY SystemZAsmPrinter : public AsmPrinter {
public:
  SystemZAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  // Override AsmPrinter.
  StringRef getPassName() const override { return "SystemZ Assembly Printer"; }
  void EmitInstruction(const MachineInstr *MI) override;
  void EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &OS) override;
};
} // end namespace llvm

#endif
