//=- X86InstComments.h - Generate verbose-asm comments for instrs -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines functionality used to emit comments about X86 instructions to
// an output stream for -fverbose-asm.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_INSTPRINTER_X86INSTCOMMENTS_H
#define LLVM_LIB_TARGET_X86_INSTPRINTER_X86INSTCOMMENTS_H

#include "llvm/CodeGen/MachineInstr.h"

namespace llvm {

  enum AsmComments {
    // For instr that was compressed from EVEX to VEX.
    AC_EVEX_2_VEX = MachineInstr::TAsmComments
  };

  class MCInst;
  class raw_ostream;
  bool EmitAnyX86InstComments(const MCInst *MI, raw_ostream &OS,
                              const char *(*getRegName)(unsigned));
}

#endif
