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

#ifndef LLVM_LIB_TARGET_MIPS_MIPS_H
#define LLVM_LIB_TARGET_MIPS_MIPS_H

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class MipsTargetMachine;
  class ModulePass;
  class FunctionPass;

  ModulePass *createMipsOs16Pass();
  ModulePass *createMips16HardFloatPass();

  FunctionPass *createMipsModuleISelDagPass();
  FunctionPass *createMipsOptimizePICCallPass();
  FunctionPass *createMipsDelaySlotFillerPass();
  FunctionPass *createMipsHazardSchedule();
  FunctionPass *createMipsLongBranchPass();
  FunctionPass *createMipsConstantIslandPass();
  FunctionPass *createMicroMipsSizeReductionPass();
} // end namespace llvm;

#endif
