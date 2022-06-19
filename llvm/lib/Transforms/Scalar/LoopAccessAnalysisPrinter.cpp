//===- LoopAccessAnalysisPrinter.cpp - Loop Access Analysis Printer --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopAccessAnalysisPrinter.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
using namespace llvm;

#define DEBUG_TYPE "loop-accesses"

PreservedAnalyses
LoopAccessInfoPrinterPass::run(Loop &L, LoopAnalysisManager &AM,
                               LoopStandardAnalysisResults &AR, LPMUpdater &) {
  Function &F = *L.getHeader()->getParent();
  auto &LAI = AM.getResult<LoopAccessAnalysis>(L, AR);
  OS << "Loop access info in function '" << F.getName() << "':\n";
  OS.indent(2) << L.getHeader()->getName() << ":\n";
  LAI.print(OS, 4);
  return PreservedAnalyses::all();
}
