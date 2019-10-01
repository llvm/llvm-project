//===-- WebAssembly.h - Top-level interface for WebAssembly  ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the entry points for global functions defined in
/// the LLVM WebAssembly back-end.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLY_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLY_H

#include "llvm/PassRegistry.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

class WebAssemblyTargetMachine;
class ModulePass;
class FunctionPass;

// LLVM IR passes.
ModulePass *createWebAssemblyLowerEmscriptenEHSjLj(bool DoEH, bool DoSjLj);
ModulePass *createWebAssemblyLowerGlobalDtors();
ModulePass *createWebAssemblyAddMissingPrototypes();
ModulePass *createWebAssemblyFixFunctionBitcasts();
FunctionPass *createWebAssemblyOptimizeReturned();

// ISel and immediate followup passes.
FunctionPass *createWebAssemblyISelDag(WebAssemblyTargetMachine &TM,
                                       CodeGenOpt::Level OptLevel);
FunctionPass *createWebAssemblyArgumentMove();
FunctionPass *createWebAssemblySetP2AlignOperands();

// Late passes.
FunctionPass *createWebAssemblyReplacePhysRegs();
FunctionPass *createWebAssemblyPrepareForLiveIntervals();
FunctionPass *createWebAssemblyOptimizeLiveIntervals();
FunctionPass *createWebAssemblyMemIntrinsicResults();
FunctionPass *createWebAssemblyRegStackify();
FunctionPass *createWebAssemblyRegColoring();
FunctionPass *createWebAssemblyFixIrreducibleControlFlow();
FunctionPass *createWebAssemblyLateEHPrepare();
FunctionPass *createWebAssemblyCFGSort();
FunctionPass *createWebAssemblyCFGStackify();
FunctionPass *createWebAssemblyExplicitLocals();
FunctionPass *createWebAssemblyLowerBrUnless();
FunctionPass *createWebAssemblyRegNumbering();
FunctionPass *createWebAssemblyPeephole();
FunctionPass *createWebAssemblyCallIndirectFixup();

// PassRegistry initialization declarations.
void initializeWebAssemblyAddMissingPrototypesPass(PassRegistry &);
void initializeWebAssemblyLowerEmscriptenEHSjLjPass(PassRegistry &);
void initializeLowerGlobalDtorsPass(PassRegistry &);
void initializeFixFunctionBitcastsPass(PassRegistry &);
void initializeOptimizeReturnedPass(PassRegistry &);
void initializeWebAssemblyArgumentMovePass(PassRegistry &);
void initializeWebAssemblySetP2AlignOperandsPass(PassRegistry &);
void initializeWebAssemblyReplacePhysRegsPass(PassRegistry &);
void initializeWebAssemblyPrepareForLiveIntervalsPass(PassRegistry &);
void initializeWebAssemblyOptimizeLiveIntervalsPass(PassRegistry &);
void initializeWebAssemblyMemIntrinsicResultsPass(PassRegistry &);
void initializeWebAssemblyRegStackifyPass(PassRegistry &);
void initializeWebAssemblyRegColoringPass(PassRegistry &);
void initializeWebAssemblyFixIrreducibleControlFlowPass(PassRegistry &);
void initializeWebAssemblyLateEHPreparePass(PassRegistry &);
void initializeWebAssemblyExceptionInfoPass(PassRegistry &);
void initializeWebAssemblyCFGSortPass(PassRegistry &);
void initializeWebAssemblyCFGStackifyPass(PassRegistry &);
void initializeWebAssemblyExplicitLocalsPass(PassRegistry &);
void initializeWebAssemblyLowerBrUnlessPass(PassRegistry &);
void initializeWebAssemblyRegNumberingPass(PassRegistry &);
void initializeWebAssemblyPeepholePass(PassRegistry &);
void initializeWebAssemblyCallIndirectFixupPass(PassRegistry &);

} // end namespace llvm

#endif
