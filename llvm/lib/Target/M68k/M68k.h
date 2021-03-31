//===- M68k.h - Top-level interface for M68k representation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the entry points for global functions defined in the
/// M68k target library, as used by the LLVM JIT.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M68K_M68K_H
#define LLVM_LIB_TARGET_M68K_M68K_H

namespace llvm {

class FunctionPass;
class M68kTargetMachine;

/// This pass converts a legalized DAG into a M68k-specific DAG, ready for
/// instruction scheduling.
FunctionPass *createM68kISelDag(M68kTargetMachine &TM);

/// Return a Machine IR pass that expands M68k-specific pseudo
/// instructions into a sequence of actual instructions. This pass
/// must run after prologue/epilogue insertion and before lowering
/// the MachineInstr to MC.
FunctionPass *createM68kExpandPseudoPass();

/// This pass initializes a global base register for PIC on M68k.
FunctionPass *createM68kGlobalBaseRegPass();

/// Finds sequential MOVEM instruction and collapse them into a single one. This
/// pass has to be run after all pseudo expansions and prologue/epilogue
/// emission so that all possible MOVEM are already in place.
FunctionPass *createM68kCollapseMOVEMPass();

/// Finds MOVE instructions before any conditioanl branch instruction and
/// replaces them with MOVEM instruction. Motorola's MOVEs do trash(V,C) flags
/// register which prevents branch from taking the correct route. This pass
/// has to be run after all pseudo expansions and prologue/epilogue emission
/// so that all possible MOVEs are present.
FunctionPass *createM68kConvertMOVToMOVMPass();

} // namespace llvm

#endif
