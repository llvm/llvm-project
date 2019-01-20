//===- llvm/Analysis/LoopAccessAnalysisPrinter.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPACCESSANALYSISPRINTER_H
#define LLVM_TRANSFORMS_SCALAR_LOOPACCESSANALYSISPRINTER_H

#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {

/// Printer pass for the \c LoopAccessInfo results.
class LoopAccessInfoPrinterPass
    : public PassInfoMixin<LoopAccessInfoPrinterPass> {
  raw_ostream &OS;

public:
  explicit LoopAccessInfoPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};

} // End llvm namespace

#endif
