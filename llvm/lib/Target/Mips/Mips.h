//===-- Mips.h - Top-level interface for Mips representation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM Mips back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_MIPS_H
#define TARGET_MIPS_H

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class MipsTargetMachine;
  class FunctionPass;

  FunctionPass *createMipsISelDag(MipsTargetMachine &TM);
  FunctionPass *createMipsDelaySlotFillerPass(MipsTargetMachine &TM);
  FunctionPass *createMipsLongBranchPass(MipsTargetMachine &TM);
  FunctionPass *createMipsJITCodeEmitterPass(MipsTargetMachine &TM,
                                             JITCodeEmitter &JCE);

} // end namespace llvm;

#endif
