//===-- X86AsmPrinter.cpp - Convert X86 LLVM IR to X86 assembly -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file the shared super class printer that converts from our internal
// representation of machine-dependent LLVM code to Intel and AT&T format
// assembly language.
// This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#include "X86ATTAsmPrinter.h"
#include "X86IntelAsmPrinter.h"
#include "X86Subtarget.h"
using namespace llvm;

/// createX86CodePrinterPass - Returns a pass that prints the X86 assembly code
/// for a MachineFunction to the given output stream, using the given target
/// machine description.
///
FunctionPass *llvm::createX86CodePrinterPass(raw_ostream &o,
                                             X86TargetMachine &tm) {
  const X86Subtarget *Subtarget = &tm.getSubtarget<X86Subtarget>();

  if (Subtarget->isFlavorIntel()) {
    return new X86IntelAsmPrinter(o, tm, tm.getTargetAsmInfo());
  } else {
    return new X86ATTAsmPrinter(o, tm, tm.getTargetAsmInfo());
  }
}

namespace {
  static struct Register {
    Register() {
      X86TargetMachine::registerAsmPrinter(createX86CodePrinterPass);
    }
  } Registrator;
}

extern "C" int X86AsmPrinterForceLink;
int X86AsmPrinterForceLink = 0;
