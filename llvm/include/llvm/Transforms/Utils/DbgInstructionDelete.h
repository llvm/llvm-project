#ifndef LLVM_TRANSFORMS_DBGINSTRUCTIONDELETE_H
#define LLVM_TRANSFORMS_DBGINSTRUCTIONDELETE_H

#include "llvm/IR/PassManager.h"

namespace llvm{

    class DbgInstructionDeletePass: public PassInfoMixin<DbgInstructionDeletePass>
    {
        public:
            PreservedAnalyses run(Function& F, FunctionAnalysisManager& AM);
    };

}


#endif