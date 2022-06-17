//
// Created by tanmay on 6/12/22.
//

#ifndef LLVM_ERRORANALYSIS_H
#define LLVM_ERRORANALYSIS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class ErrorAnalysisPass : public PassInfoMixin<ErrorAnalysisPass>{
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_ERRORANALYSIS_H
