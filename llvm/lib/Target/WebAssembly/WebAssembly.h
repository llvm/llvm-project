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

#include "GISel/WebAssemblyRegisterBankInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {

class WebAssemblyTargetMachine;
class ModulePass;
class FunctionPass;

// LLVM IR passes.
class WebAssemblyLowerEmscriptenEHSjLjPass
    : public RequiredPassInfoMixin<WebAssemblyLowerEmscriptenEHSjLjPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createWebAssemblyLowerEmscriptenEHSjLjLegacyPass();

class WebAssemblyAddMissingPrototypesPass
    : public RequiredPassInfoMixin<WebAssemblyAddMissingPrototypesPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createWebAssemblyAddMissingPrototypesLegacyPass();

class WebAssemblyFixFunctionBitcastsPass
    : public RequiredPassInfoMixin<WebAssemblyFixFunctionBitcastsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createWebAssemblyFixFunctionBitcastsLegacyPass();

class WebAssemblyOptimizeReturnedPass
    : public OptionalPassInfoMixin<WebAssemblyOptimizeReturnedPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

FunctionPass *createWebAssemblyOptimizeReturnedLegacyPass();

class WebAssemblyRefTypeMem2LocalPass
    : public RequiredPassInfoMixin<WebAssemblyRefTypeMem2LocalPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

FunctionPass *createWebAssemblyRefTypeMem2LocalLegacyPass();

class WebAssemblyReduceToAnyAllTruePass
    : public RequiredPassInfoMixin<WebAssemblyReduceToAnyAllTruePass> {
private:
  Module *CachedModule = nullptr;
  bool ModuleHasInterestingIntrinsics = false;
  WebAssemblyTargetMachine &TM;

public:
  WebAssemblyReduceToAnyAllTruePass(WebAssemblyTargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

FunctionPass *
createWebAssemblyReduceToAnyAllTrueLegacyPass(WebAssemblyTargetMachine &TM);

class WebAssemblyCoalesceFeaturesAndStripAtomicsPass
    : public RequiredPassInfoMixin<
          WebAssemblyCoalesceFeaturesAndStripAtomicsPass> {
  WebAssemblyTargetMachine &TM;

public:
  WebAssemblyCoalesceFeaturesAndStripAtomicsPass(WebAssemblyTargetMachine &TM)
      : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createWebAssemblyCoalesceFeaturesAndStripAtomicsLegacyPass(
    WebAssemblyTargetMachine &TM);

// GlobalISel
InstructionSelector *
createWebAssemblyInstructionSelector(const WebAssemblyTargetMachine &,
                                     const WebAssemblySubtarget &,
                                     const WebAssemblyRegisterBankInfo &);

FunctionPass *createWebAssemblyPostLegalizerCombiner();
void initializeWebAssemblyPostLegalizerCombinerPass(PassRegistry &);

FunctionPass *createWebAssemblyPreLegalizerCombiner();
void initializeWebAssemblyPreLegalizerCombinerPass(PassRegistry &);

// ISel and immediate followup passes.
class WebAssemblyISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  WebAssemblyISelDAGToDAGPass(WebAssemblyTargetMachine &TM,
                              CodeGenOptLevel OptLevel);
};

FunctionPass *createWebAssemblyISelDagLegacyPass(WebAssemblyTargetMachine &TM,
                                                 CodeGenOptLevel OptLevel);

class WebAssemblyArgumentMovePass
    : public RequiredPassInfoMixin<WebAssemblyArgumentMovePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createWebAssemblyArgumentMoveLegacyPass();

class WebAssemblySetP2AlignOperandsPass
    : public RequiredPassInfoMixin<WebAssemblySetP2AlignOperandsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createWebAssemblySetP2AlignOperandsLegacyPass();

class WebAssemblyCleanCodeAfterTrapPass
    : public RequiredPassInfoMixin<WebAssemblyCleanCodeAfterTrapPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createWebAssemblyCleanCodeAfterTrapLegacyPass();

// Late passes.
class WebAssemblyReplacePhysRegsPass
    : public RequiredPassInfoMixin<WebAssemblyReplacePhysRegsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createWebAssemblyReplacePhysRegsLegacyPass();

class WebAssemblyNullifyDebugValueListsPass
    : public RequiredPassInfoMixin<WebAssemblyNullifyDebugValueListsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createWebAssemblyNullifyDebugValueListsLegacyPass();
FunctionPass *createWebAssemblyOptimizeLiveIntervals();
FunctionPass *createWebAssemblyMemIntrinsicResults();
FunctionPass *createWebAssemblyRegStackify(CodeGenOptLevel OptLevel);
FunctionPass *createWebAssemblyRegColoring();

class WebAssemblyFixBrTableDefaultsPass
    : public RequiredPassInfoMixin<WebAssemblyFixBrTableDefaultsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createWebAssemblyFixBrTableDefaultsLegacyPass();

class WebAssemblyFixIrreducibleControlFlowPass
    : public RequiredPassInfoMixin<WebAssemblyFixIrreducibleControlFlowPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createWebAssemblyFixIrreducibleControlFlowLegacyPass();

class WebAssemblyLateEHPreparePass
    : public RequiredPassInfoMixin<WebAssemblyLateEHPreparePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createWebAssemblyLateEHPrepareLegacyPass();
FunctionPass *createWebAssemblyCFGSort();
FunctionPass *createWebAssemblyCFGStackify();
FunctionPass *createWebAssemblyExplicitLocals();
FunctionPass *createWebAssemblyLowerBrUnless();
FunctionPass *createWebAssemblyRegNumbering();
FunctionPass *createWebAssemblyVecReduce();
FunctionPass *createWebAssemblyDebugFixup();
FunctionPass *createWebAssemblyPeephole();
ModulePass *createWebAssemblyMCLowerPrePass();

// PassRegistry initialization declarations.
void initializeWebAssemblyOptimizeReturnedLegacyPass(PassRegistry &);
void initializeWebAssemblyRefTypeMem2LocalLegacyPass(PassRegistry &);
void initializeWebAssemblyAddMissingPrototypesLegacyPass(PassRegistry &);
void initializeWebAssemblyArgumentMoveLegacyPass(PassRegistry &);
void initializeWebAssemblyAsmPrinterPass(PassRegistry &);
void initializeWebAssemblyCleanCodeAfterTrapLegacyPass(PassRegistry &);
void initializeWebAssemblyCFGSortPass(PassRegistry &);
void initializeWebAssemblyCFGStackifyPass(PassRegistry &);
void initializeWebAssemblyDAGToDAGISelLegacyPass(PassRegistry &);
void initializeWebAssemblyDebugFixupPass(PassRegistry &);
void initializeWebAssemblyExceptionInfoPass(PassRegistry &);
void initializeWebAssemblyExplicitLocalsPass(PassRegistry &);
void initializeWebAssemblyFixBrTableDefaultsLegacyPass(PassRegistry &);
void initializeWebAssemblyFixFunctionBitcastsLegacyPass(PassRegistry &);
void initializeWebAssemblyFixIrreducibleControlFlowLegacyPass(PassRegistry &);
void initializeWebAssemblyLateEHPrepareLegacyPass(PassRegistry &);
void initializeWebAssemblyLowerBrUnlessPass(PassRegistry &);
void initializeWebAssemblyLowerEmscriptenEHSjLjLegacyPass(PassRegistry &);
void initializeWebAssemblyMCLowerPrePassPass(PassRegistry &);
void initializeWebAssemblyMemIntrinsicResultsPass(PassRegistry &);
void initializeWebAssemblyNullifyDebugValueListsLegacyPass(PassRegistry &);
void initializeWebAssemblyOptimizeLiveIntervalsPass(PassRegistry &);
void initializeWebAssemblyPeepholePass(PassRegistry &);
void initializeWebAssemblyRegColoringPass(PassRegistry &);
void initializeWebAssemblyRegNumberingPass(PassRegistry &);
void initializeWebAssemblyRegStackifyPass(PassRegistry &);
void initializeWebAssemblyReplacePhysRegsLegacyPass(PassRegistry &);
void initializeWebAssemblySetP2AlignOperandsLegacyPass(PassRegistry &);
void initializeWebAssemblyCoalesceFeaturesAndStripAtomicsLegacyPass(
    PassRegistry &);

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
