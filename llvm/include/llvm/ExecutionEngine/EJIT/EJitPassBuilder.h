//===-- EJitPassBuilder.h - Minimal PassBuilder for EJIT -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A minimal replacement for llvm::PassBuilder that registers only the analyses
// needed by the EJIT JIT optimization pipeline (8 passes).  This avoids linking
// the full Passes component (19 LINK_COMPONENTS).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITPASSBUILDER_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITPASSBUILDER_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"

namespace llvm {
namespace ejit {

/// Registers only the analyses required by EJIT's optimization pipeline.
///
/// EJIT uses 8 passes: InstCombine, SCCP, ADCE, SimplifyCFG,
/// LoopFullUnroll, LoopSimplify, AlwaysInliner, Promote (Mem2Reg).
///
/// PassBuilder::registerFunctionAnalyses registers ~40 analyses; we register
/// only the ~13 that these 8 passes actually need.
namespace EJitPassBuilder {

void registerFunctionAnalyses(FunctionAnalysisManager &FAM);
void registerLoopAnalyses(LoopAnalysisManager &LAM);
void registerCGSCCAnalyses(CGSCCAnalysisManager &CGAM);
void registerModuleAnalyses(ModuleAnalysisManager &MAM);

void crossRegisterProxies(LoopAnalysisManager &LAM,
                          FunctionAnalysisManager &FAM,
                          CGSCCAnalysisManager &CGAM,
                          ModuleAnalysisManager &MAM);

} // namespace EJitPassBuilder
} // namespace ejit
} // namespace llvm

#endif
