//===-- MipsDirectObjLower.h - Mips LLVM direct object lowering *- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSDIRECTOBJLOWER_H
#define MIPSDIRECTOBJLOWER_H
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
  class MCInst;
  class MCStreamer;

  namespace Mips {
  /// MipsDirectObjLower - This name space is used to lower MCInstr in cases
  //                       where the assembler usually finishes the lowering
  //                       such as large shifts.
    void LowerLargeShift(MCInst &Inst);
    void LowerDextDins(MCInst &Inst);
  }
}

#endif
