#ifndef LLVM_TRANSFORMS_IPO_NOINLINEFUNCCALLEDONCE_H
#define LLVM_TRANSFORMS_IPO_NOINLINEFUNCCALLEDONCE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

struct NoInlineFuncCalledOncePass
    : public PassInfoMixin<NoInlineFuncCalledOncePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

// single definition of the control flag lives in the .cpp (declared here)
extern cl::opt<bool> EnableNoInlineFuncCalledOnce;

} // namespace llvm
#endif
