//
// Created by tanmay on 6/6/22.
//

#ifndef LLVM_COMPUTATIONGRAPHGENERATOR_H
#define LLVM_COMPUTATIONGRAPHGENERATOR_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Analysis/ComputationGraph.h"

namespace llvm {

class ComputationGraphAnalysis : public AnalysisInfoMixin<ComputationGraphAnalysis> {

  friend AnalysisInfoMixin<ComputationGraphAnalysis>;

  static AnalysisKey Key;
public:
  using Result = ComputationGraph;

  ComputationGraph run(Function &F, FunctionAnalysisManager &AM);
};

// Printer Pass for the ComputationGraphPathAnalysis results
class ComputationGraphPrinterPass
    : public PassInfoMixin<ComputationGraphPrinterPass> {
  raw_ostream &OS;

public:
  explicit ComputationGraphPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_COMPUTATIONGRAPHGENERATOR_H
