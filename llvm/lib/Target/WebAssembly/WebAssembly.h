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
FunctionPass *createWebAssemblyNullifyDebugValueLists();
FunctionPass *createWebAssemblyPrepareForLiveIntervals();
FunctionPass *createWebAssemblyOptimizeLiveIntervals();
FunctionPass *createWebAssemblyMemIntrinsicResults();
FunctionPass *createWebAssemblyRegStackify();
FunctionPass *createWebAssemblyRegColoring();
FunctionPass *createWebAssemblyFixBrTableDefaults();
FunctionPass *createWebAssemblyFixIrreducibleControlFlow();
FunctionPass *createWebAssemblyLateEHPrepare();
FunctionPass *createWebAssemblyCFGSort();
FunctionPass *createWebAssemblyCFGStackify();
FunctionPass *createWebAssemblyExplicitLocals();
FunctionPass *createWebAssemblyLowerBrUnless();
FunctionPass *createWebAssemblyRegNumbering();
FunctionPass *createWebAssemblyDebugFixup();
FunctionPass *createWebAssemblyPeephole();

// PassRegistry initialization declarations.
void initializeWebAssemblyAddMissingPrototypesPass(PassRegistry &);
void initializeWebAssemblyLowerEmscriptenEHSjLjPass(PassRegistry &);
void initializeLowerGlobalDtorsPass(PassRegistry &);
void initializeFixFunctionBitcastsPass(PassRegistry &);
void initializeOptimizeReturnedPass(PassRegistry &);
void initializeWebAssemblyArgumentMovePass(PassRegistry &);
void initializeWebAssemblySetP2AlignOperandsPass(PassRegistry &);
void initializeWebAssemblyReplacePhysRegsPass(PassRegistry &);
void initializeWebAssemblyNullifyDebugValueListsPass(PassRegistry &);
void initializeWebAssemblyPrepareForLiveIntervalsPass(PassRegistry &);
void initializeWebAssemblyOptimizeLiveIntervalsPass(PassRegistry &);
void initializeWebAssemblyMemIntrinsicResultsPass(PassRegistry &);
void initializeWebAssemblyRegStackifyPass(PassRegistry &);
void initializeWebAssemblyRegColoringPass(PassRegistry &);
void initializeWebAssemblyFixBrTableDefaultsPass(PassRegistry &);
void initializeWebAssemblyFixIrreducibleControlFlowPass(PassRegistry &);
void initializeWebAssemblyLateEHPreparePass(PassRegistry &);
void initializeWebAssemblyExceptionInfoPass(PassRegistry &);
void initializeWebAssemblyCFGSortPass(PassRegistry &);
void initializeWebAssemblyCFGStackifyPass(PassRegistry &);
void initializeWebAssemblyExplicitLocalsPass(PassRegistry &);
void initializeWebAssemblyLowerBrUnlessPass(PassRegistry &);
void initializeWebAssemblyRegNumberingPass(PassRegistry &);
void initializeWebAssemblyDebugFixupPass(PassRegistry &);
void initializeWebAssemblyPeepholePass(PassRegistry &);

namespace WebAssembly {
enum TargetIndex {
  // Followed by a local index (ULEB).
  TI_LOCAL,
  // Followed by an absolute global index (ULEB). DEPRECATED.
  TI_GLOBAL_FIXED,
  // Followed by the index from the bottom of the Wasm stack.
  TI_OPERAND_STACK,
  // Followed by a compilation unit relative global index (uint32_t)
  // that will have an associated relocation.
  TI_GLOBAL_RELOC,
  // Like TI_LOCAL, but indicates an indirect value (e.g. byval arg
  // passed by pointer).
  TI_LOCAL_INDIRECT
};
} // end namespace WebAssembly

} // end namespace llvm

#endif
