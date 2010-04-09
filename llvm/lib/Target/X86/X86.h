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

#include "llvm/Target/TargetMachine.h"

namespace llvm {

class FunctionPass;
class JITCodeEmitter;
class MCAssembler;
class MCCodeEmitter;
class MCContext;
class MachineCodeEmitter;
class Target;
class TargetAsmBackend;
class X86TargetMachine;
class formatted_raw_ostream;

/// createX86ISelDag - This pass converts a legalized DAG into a 
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *createX86ISelDag(X86TargetMachine &TM,
                               CodeGenOpt::Level OptLevel);

/// createX86FloatingPointStackifierPass - This function returns a pass which
/// converts floating point register references and pseudo instructions into
/// floating point stack references and physical instructions.
///
FunctionPass *createX86FloatingPointStackifierPass();

/// createX87FPRegKillInserterPass - This function returns a pass which
/// inserts FP_REG_KILL instructions where needed.
///
FunctionPass *createX87FPRegKillInserterPass();

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *createX86JITCodeEmitterPass(X86TargetMachine &TM,
                                          JITCodeEmitter &JCE);

MCCodeEmitter *createX86_32MCCodeEmitter(const Target &, TargetMachine &TM,
                                         MCContext &Ctx);
MCCodeEmitter *createX86_64MCCodeEmitter(const Target &, TargetMachine &TM,
                                         MCContext &Ctx);

TargetAsmBackend *createX86_32AsmBackend(const Target &, MCAssembler &);
TargetAsmBackend *createX86_64AsmBackend(const Target &, MCAssembler &);

/// createX86EmitCodeToMemory - Returns a pass that converts a register
/// allocated function into raw machine code in a dynamically
/// allocated chunk of memory.
///
FunctionPass *createEmitX86CodeToMemory();

/// createX86MaxStackAlignmentHeuristicPass - This function returns a pass
/// which determines whether the frame pointer register should be
/// reserved in case dynamic stack alignment is later required.
///
FunctionPass *createX86MaxStackAlignmentHeuristicPass();

extern Target TheX86_32Target, TheX86_64Target;

} // End llvm namespace

// Defines symbolic names for X86 registers.  This defines a mapping from
// register name to register number.
//
#include "X86GenRegisterNames.inc"

// Defines symbolic names for the X86 instructions.
//
#include "X86GenInstrNames.inc"

#endif
