#ifndef POLLY_CODEPREPARATION_H
#define POLLY_CODEPREPARATION_H

#include "llvm/IR/PassManager.h"

namespace polly {
struct CodePreparationPass : public llvm::PassInfoMixin<CodePreparationPass> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
} // namespace polly

#endif /* POLLY_CODEPREPARATION_H */
