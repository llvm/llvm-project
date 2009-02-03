//===-- Alpha.h - Top-level interface for Alpha representation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// Alpha back-end.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_ALPHA_H
#define TARGET_ALPHA_H

namespace llvm {

  class AlphaTargetMachine;
  class FunctionPass;
  class TargetMachine;
  class MachineCodeEmitter;
  class raw_ostream;

  FunctionPass *createAlphaISelDag(AlphaTargetMachine &TM);
  FunctionPass *createAlphaCodePrinterPass(raw_ostream &OS,
                                             TargetMachine &TM);
  FunctionPass *createAlphaPatternInstructionSelector(TargetMachine &TM);
  FunctionPass *createAlphaCodeEmitterPass(AlphaTargetMachine &TM,
                                           MachineCodeEmitter &MCE);
  FunctionPass *createAlphaLLRPPass(AlphaTargetMachine &tm);
  FunctionPass *createAlphaBranchSelectionPass();

} // end namespace llvm;

// Defines symbolic names for Alpha registers.  This defines a mapping from
// register name to register number.
//
#include "AlphaGenRegisterNames.inc"

// Defines symbolic names for the Alpha instructions.
//
#include "AlphaGenInstrNames.inc"

#endif
