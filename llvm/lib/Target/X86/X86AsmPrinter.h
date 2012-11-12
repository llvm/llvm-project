//===-- X86AsmPrinter.h - Convert X86 LLVM code to assembly -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AT&T assembly code printer class.
//
//===----------------------------------------------------------------------===//

#ifndef X86ASMPRINTER_H
#define X86ASMPRINTER_H

#include "X86.h"
#include "X86MachineFunctionInfo.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MCStreamer;

class LLVM_LIBRARY_VISIBILITY X86AsmPrinter : public AsmPrinter {
  const X86Subtarget *Subtarget;
 public:
  explicit X86AsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer) {
    Subtarget = &TM.getSubtarget<X86Subtarget>();
  }

  virtual const char *getPassName() const LLVM_OVERRIDE {
    return "X86 AT&T-Style Assembly Printer";
  }

  const X86Subtarget &getSubtarget() const { return *Subtarget; }

  virtual void EmitStartOfAsmFile(Module &M) LLVM_OVERRIDE;

  virtual void EmitEndOfAsmFile(Module &M) LLVM_OVERRIDE;

  virtual void EmitInstruction(const MachineInstr *MI) LLVM_OVERRIDE;

  void printSymbolOperand(const MachineOperand &MO, raw_ostream &O);

  // These methods are used by the tablegen'erated instruction printer.
  void printOperand(const MachineInstr *MI, unsigned OpNo, raw_ostream &O,
                    const char *Modifier = 0, unsigned AsmVariant = 0);
  void printPCRelImm(const MachineInstr *MI, unsigned OpNo, raw_ostream &O);

  bool printAsmMRegister(const MachineOperand &MO, char Mode, raw_ostream &O);
  virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &OS) LLVM_OVERRIDE;
  virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                     unsigned AsmVariant, const char *ExtraCode,
                                     raw_ostream &OS) LLVM_OVERRIDE;

  void printMemReference(const MachineInstr *MI, unsigned Op, raw_ostream &O,
                         const char *Modifier=NULL);
  void printLeaMemReference(const MachineInstr *MI, unsigned Op, raw_ostream &O,
                            const char *Modifier=NULL);

  void printIntelMemReference(const MachineInstr *MI, unsigned Op,
                              raw_ostream &O, const char *Modifier=NULL,
                              unsigned AsmVariant = 1);

  virtual bool runOnMachineFunction(MachineFunction &F) LLVM_OVERRIDE;

  void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);

  virtual MachineLocation
    getDebugValueLocation(const MachineInstr *MI) const LLVM_OVERRIDE;
};

} // end namespace llvm

#endif
