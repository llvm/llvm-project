//===-- X86.h - Top-level interface for X86 representation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the x86
// target library, as used by the LLVM JIT.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_X86_H
#define TARGET_X86_H

namespace llvm {

class X86TargetMachine;
class FunctionPass;
class MachineCodeEmitter;
class raw_ostream;

/// createX86ISelDag - This pass converts a legalized DAG into a 
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *createX86ISelDag(X86TargetMachine &TM, bool Fast);

/// createX86FloatingPointStackifierPass - This function returns a pass which
/// converts floating point register references and pseudo instructions into
/// floating point stack references and physical instructions.
///
FunctionPass *createX86FloatingPointStackifierPass();

/// createX87FPRegKillInserterPass - This function returns a pass which
/// inserts FP_REG_KILL instructions where needed.
///
FunctionPass *createX87FPRegKillInserterPass();

/// createX86CodePrinterPass - Returns a pass that prints the X86
/// assembly code for a MachineFunction to the given output stream,
/// using the given target machine description.
///
FunctionPass *createX86CodePrinterPass(raw_ostream &o, X86TargetMachine &tm);

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *createX86CodeEmitterPass(X86TargetMachine &TM,
                                       MachineCodeEmitter &MCE);

/// createX86EmitCodeToMemory - Returns a pass that converts a register
/// allocated function into raw machine code in a dynamically
/// allocated chunk of memory.
///
FunctionPass *createEmitX86CodeToMemory();

/// createX86MaxStackAlignmentCalculatorPass - This function returns a pass which
/// calculates maximal stack alignment required for function
///
FunctionPass *createX86MaxStackAlignmentCalculatorPass();

} // End llvm namespace

// Defines symbolic names for X86 registers.  This defines a mapping from
// register name to register number.
//
#include "X86GenRegisterNames.inc"

// Defines symbolic names for the X86 instructions.
//
#include "X86GenInstrNames.inc"

#endif
