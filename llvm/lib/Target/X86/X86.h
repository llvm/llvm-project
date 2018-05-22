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

#ifndef LLVM_LIB_TARGET_X86_X86_H
#define LLVM_LIB_TARGET_X86_X86_H

#include "llvm/Support/CodeGen.h"

namespace llvm {

class FunctionPass;
class ImmutablePass;
class InstructionSelector;
class ModulePass;
class PassRegistry;
class X86RegisterBankInfo;
class X86Subtarget;
class X86TargetMachine;

/// This pass converts a legalized DAG into a X86-specific DAG, ready for
/// instruction scheduling.
FunctionPass *createX86ISelDag(X86TargetMachine &TM,
                               CodeGenOpt::Level OptLevel);

/// This pass initializes a global base register for PIC on x86-32.
FunctionPass *createX86GlobalBaseRegPass();

/// This pass combines multiple accesses to local-dynamic TLS variables so that
/// the TLS base address for the module is only fetched once per execution path
/// through the function.
FunctionPass *createCleanupLocalDynamicTLSPass();

/// This function returns a pass which converts floating-point register
/// references and pseudo instructions into floating-point stack references and
/// physical instructions.
FunctionPass *createX86FloatingPointStackifierPass();

/// This pass inserts AVX vzeroupper instructions before each call to avoid
/// transition penalty between functions encoded with AVX and SSE.
FunctionPass *createX86IssueVZeroUpperPass();

/// Return a pass that pads short functions with NOOPs.
/// This will prevent a stall when returning on the Atom.
FunctionPass *createX86PadShortFunctions();

/// Return a pass that selectively replaces certain instructions (like add,
/// sub, inc, dec, some shifts, and some multiplies) by equivalent LEA
/// instructions, in order to eliminate execution delays in some processors.
FunctionPass *createX86FixupLEAs();

/// Return a pass that removes redundant LEA instructions and redundant address
/// recalculations.
FunctionPass *createX86OptimizeLEAs();

/// Return a pass that transforms setcc + movzx pairs into xor + setcc.
FunctionPass *createX86FixupSetCC();

/// Return a pass that lowers EFLAGS copy pseudo instructions.
FunctionPass *createX86FlagsCopyLoweringPass();

/// Return a pass that expands WinAlloca pseudo-instructions.
FunctionPass *createX86WinAllocaExpander();

/// Return a pass that optimizes the code-size of x86 call sequences. This is
/// done by replacing esp-relative movs with pushes.
FunctionPass *createX86CallFrameOptimization();

/// Return an IR pass that inserts EH registration stack objects and explicit
/// EH state updates. This pass must run after EH preparation, which does
/// Windows-specific but architecture-neutral preparation.
FunctionPass *createX86WinEHStatePass();

/// Return a Machine IR pass that expands X86-specific pseudo
/// instructions into a sequence of actual instructions. This pass
/// must run after prologue/epilogue insertion and before lowering
/// the MachineInstr to MC.
FunctionPass *createX86ExpandPseudoPass();

/// This pass converts X86 cmov instructions into branch when profitable.
FunctionPass *createX86CmovConverterPass();

/// Return a Machine IR pass that selectively replaces
/// certain byte and word instructions by equivalent 32 bit instructions,
/// in order to eliminate partial register usage, false dependences on
/// the upper portions of registers, and to save code size.
FunctionPass *createX86FixupBWInsts();

/// Return a Machine IR pass that reassigns instruction chains from one domain
/// to another, when profitable.
FunctionPass *createX86DomainReassignmentPass();

void initializeFixupBWInstPassPass(PassRegistry &);

/// This pass replaces EVEX encoded of AVX-512 instructiosn by VEX
/// encoding when possible in order to reduce code size.
FunctionPass *createX86EvexToVexInsts();

/// This pass creates the thunks for the retpoline feature.
FunctionPass *createX86RetpolineThunksPass();

InstructionSelector *createX86InstructionSelector(const X86TargetMachine &TM,
                                                  X86Subtarget &,
                                                  X86RegisterBankInfo &);

void initializeEvexToVexInstPassPass(PassRegistry &);

} // End llvm namespace

#endif
