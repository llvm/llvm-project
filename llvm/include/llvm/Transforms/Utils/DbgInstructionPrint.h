#ifndef LLVM_TRANSFORMS_DBGINSTRUCTIONPRINT_H
#define LLVM_TRANSFORMS_DBGINSTRUCTIONPRINT_H

#include "llvm/IR/PassManager.h"

namespace llvm{
    class DbgInstructionPrintPass : public PassInfoMixin<DbgInstructionPrintPass>
    {
        public:

        PreservedAnalyses run(Function& F, FunctionAnalysisManager& AM);
    };
}

#endif