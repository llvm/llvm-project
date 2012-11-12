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

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class FunctionPass;
class JITCodeEmitter;
class X86TargetMachine;

/// createX86ISelDag - This pass converts a legalized DAG into a
/// X86-specific DAG, ready for instruction scheduling.
///
FunctionPass *createX86ISelDag(X86TargetMachine &TM,
                               CodeGenOpt::Level OptLevel);

/// createGlobalBaseRegPass - This pass initializes a global base
/// register for PIC on x86-32.
FunctionPass* createGlobalBaseRegPass();

/// createCleanupLocalDynamicTLSPass() - This pass combines multiple accesses
/// to local-dynamic TLS variables so that the TLS base address for the module
/// is only fetched once per execution path through the function.
FunctionPass *createCleanupLocalDynamicTLSPass();

/// createX86FloatingPointStackifierPass - This function returns a pass which
/// converts floating point register references and pseudo instructions into
/// floating point stack references and physical instructions.
///
FunctionPass *createX86FloatingPointStackifierPass();

/// createX86IssueVZeroUpperPass - This pass inserts AVX vzeroupper instructions
/// before each call to avoid transition penalty between functions encoded with
/// AVX and SSE.
FunctionPass *createX86IssueVZeroUpperPass();

/// createX86CodeEmitterPass - Return a pass that emits the collected X86 code
/// to the specified MCE object.
FunctionPass *createX86JITCodeEmitterPass(X86TargetMachine &TM,
                                          JITCodeEmitter &JCE);

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

} // End llvm namespace

#endif
