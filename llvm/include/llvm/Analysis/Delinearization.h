//===---- Delinearization.h - MultiDimensional Index Delinearization ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements an analysis pass that tries to delinearize all GEP
// instructions in all loops using the SCEV analysis functionality. This pass is
// only used for testing purposes: if your pass needs delinearization, please
// use the on-demand SCEVAddRecExpr::delinearize() function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DELINEARIZATION_H
#define LLVM_ANALYSIS_DELINEARIZATION_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
struct DelinearizationPrinterPass
    : public PassInfoMixin<DelinearizationPrinterPass> {
  explicit DelinearizationPrinterPass(raw_ostream &OS);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  raw_ostream &OS;
};
} // namespace llvm

#endif // LLVM_ANALYSIS_DELINEARIZATION_H
